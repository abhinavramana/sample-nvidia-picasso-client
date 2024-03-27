import asyncio
import base64
import io
import time
import zipfile
from typing import List, Tuple, Any

import aiohttp
from sample_client_api.log_handling import get_logger_for_file

from sample_client_api import config
from sample_client_api.nvidia.client.nvidia_request import NvidiaRequest

logger = get_logger_for_file(__name__)


class NvidiaFulfilledResponseProcessingException(Exception):
    def __init__(self, req_id, e, nvidia_client_request, task_id, res_json):
        self.req_id = req_id
        self.e = e
        self.nvidia_client_request = nvidia_client_request
        self.task_id = task_id
        self.res_json = res_json
        message = f"req_id: {req_id} failed for {task_id}: {nvidia_client_request} due to reason: {e} with response: {res_json}"
        super().__init__(message)


class NvidiaImageProcessingException(NvidiaFulfilledResponseProcessingException):
    def __init__(self, req_id, e, nvidia_client_request, task_id: str, res_json):
        super().__init__(req_id, e, nvidia_client_request, task_id, res_json)


class NvidiaImageZipRetrievalException(NvidiaFulfilledResponseProcessingException):
    def __init__(
        self,
        req_id: str,
        e,
        nvidia_client_request: NvidiaRequest,
        task_id: str,
        res_json,
    ):
        super().__init__(req_id, e, nvidia_client_request, task_id, res_json)


class NvidiaPollTimeoutException(Exception):
    def __init__(self, nvidia_request: NvidiaRequest, task_id: str, req_id: str):
        message = f"Task timed out after {config.NVCF_MAX_POLLING_ATTEMPTS} attempts for Request: {task_id}: {nvidia_request} for req_id: {req_id}"
        super().__init__(message)


async def explicitly_sleep_for_minimum_polling_interval(last_request_time: float):
    if time.time() - last_request_time < config.NVCF_MIN_POLLING_INTERVAL:
        sleep_time = config.NVCF_MIN_POLLING_INTERVAL - (
            time.time() - last_request_time
        )
        logger.info("Sleeping for %s seconds", sleep_time)
        await asyncio.sleep(sleep_time)


NVIDIA_ZIP_IMAGE_FILE_NAME = "image.jpg"


async def handle_fulfilled_response(
    session: aiohttp.ClientSession,
    res_json,
    nvidia_client_request: NvidiaRequest,
    task_id: str,
) -> Tuple[bytes, List[Any]]:
    req_id = res_json["reqId"]
    outputs = res_json.get("response", {}).get("outputs", [])
    image_data: bytes

    if res_json.get("responseReference") is not None:  # zip file was sent back
        try:
            url = res_json["responseReference"]
            image_data = await convert_zipped_image_from_url_to_base64(session, url)
        except Exception as e:
            raise NvidiaImageZipRetrievalException(
                req_id, e, nvidia_client_request, task_id, res_json
            )
    else:
        try:
            image_outputs = outputs[0]
            image_base64 = image_outputs["data"][0]  # Means outputs are not empty
            image_data = await asyncio.get_running_loop().run_in_executor(
                None, lambda: base64.b64decode(image_base64)
            )
        except Exception as e:
            logger.error(f"Error getting image from response: {e}", exc_info=True)
            raise NvidiaImageProcessingException(
                req_id, e, nvidia_client_request, task_id, res_json
            )

    return image_data, [output["data"][0] for output in outputs[1:]]


async def convert_zipped_image_from_url_to_base64(
    session: aiohttp.ClientSession, url: str
) -> bytes:
    # Step 1: Download the zip file from the URL
    logger.info(f"Getting zip file from {url}...")
    async with session.get(url) as response:
        if response.status != 200:  # Make sure the request was successful
            raise ValueError(
                f"Failed to download file from {url} with status code: {response.status}"
            )

        # Step 2: Open the zip file and read the file data
        image_data = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda data: zipfile.ZipFile(io.BytesIO(data)).read(
                NVIDIA_ZIP_IMAGE_FILE_NAME
            ),
            await response.read(),
        )

        return image_data
