import json
import time
from enum import unique, Enum
from typing import Optional, Dict, Any, List, Tuple

import aiohttp
from aiohttp import ClientResponse
from sample_client_api.log_handling import get_logger_for_file

from sample_client_api import config
from sample_client_api.config import NVCF_SDXL_DIFFUSION_FUNCTION_ID
from sample_client_api.nvidia.client.nvidia_asset_client import (
    NvidiaAssetClient,
    is_response_status_valid,
)
from sample_client_api.nvidia.client.nvidia_exceptions import (
    NvidiaPollException,
    NSFWRejectionException,
    NvidiaFunctionNotFoundException,
    NvidiaPostClientException,
    NSFWRejectionFaceswapException,
    NSFWRejectionSDXLException,
    NvidiaOOMException,
)
from sample_client_api.nvidia.client.nvidia_request import (
    NvidiaRequest,
    NvidiaRequestParameter,
    FACESWAP_FUNCTION_ID_SET,
)
from sample_client_api.nvidia.client.nvidia_response_handler import (
    NvidiaPollTimeoutException,
    explicitly_sleep_for_minimum_polling_interval,
    handle_fulfilled_response,
)
from sample_client_api.nvidia.nvidia_token_manager import NvidiaAuthConfig, NvidiaAuthTokenManager

NVCF_POLL_SECONDS = "60"  # valid range is 0-300 seconds

logger = get_logger_for_file(__name__)


@unique
class NvidiaFunctionResponseStatus(Enum):
    """
    Statuses fulfilled, rejected and errored are completed states, and you should not continue to poll.
    """

    # Worker node has not yet accepted the request.
    PENDING_EVALUATION = "pending-evaluation"
    FULFILLED = "fulfilled"  # The process has been completed with results.
    REJECTED = "rejected"  # The request was rejected by the service.
    ERRORED = "errored"  # An error occurred during Worker node processing.
    IN_PROGRESS = "in-progress"  # A Worker node is processing the request.


NVIDIA_FUNCTION_POLLING_STATUSES = {
    NvidiaFunctionResponseStatus.PENDING_EVALUATION.value,
    NvidiaFunctionResponseStatus.IN_PROGRESS.value,
}


class InvalidNvidiaPollParamsException(Exception):
    pass


def check_missing_function_id(nvidia_request, task_id, status, exception_reason):
    if (
        "Specified function in account" in exception_reason
        and "is not found" in exception_reason
    ):
        raise NvidiaFunctionNotFoundException(
            nvidia_request, task_id, "", status, exception_reason
        )


def check_oom_exception(nvidia_request, task_id, status, exception_reason):
    if (
        "OutOfMemoryError" in exception_reason
        or "CUDA out of memory. Tried to allocate" in exception_reason
    ):
        raise NvidiaOOMException(nvidia_request, task_id, "", status, exception_reason)


def check_nsfw_exception(nvidia_request, task_id, status, exception_reason):
    if "nsfwrejection" in exception_reason.lower():
        if nvidia_request.function_id in FACESWAP_FUNCTION_ID_SET:
            raise NSFWRejectionFaceswapException(
                nvidia_request, task_id, "", status, exception_reason
            )
        if nvidia_request.function_id == NVCF_SDXL_DIFFUSION_FUNCTION_ID:
            raise NSFWRejectionSDXLException(
                nvidia_request, task_id, "", status, exception_reason
            )
        raise NSFWRejectionException(
            nvidia_request, task_id, "", status, exception_reason
        )


def check_custom_exception_reasons(nvidia_request, task_id, status, exception_reason):
    check_oom_exception(nvidia_request, task_id, status, exception_reason)
    check_nsfw_exception(nvidia_request, task_id, status, exception_reason)
    check_missing_function_id(nvidia_request, task_id, status, exception_reason)


def process_parameter(name: str, parameter: Any) -> Dict[str, Any]:
    if not isinstance(parameter, NvidiaRequestParameter):
        parameter = NvidiaRequestParameter(parameter)

    return {
        "name": name,
        "shape": [1],
        "datatype": parameter.detect_type(),
        "data": [parameter.value],
    }


class NvidiaImageGenerationClient:
    def __init__(self, nvcf_url: str, auth_config: NvidiaAuthConfig):
        logger.info("Initializing NvidiaImageGenerationClient...")
        self.token_manager = NvidiaAuthTokenManager(auth_config)
        self.endpoint = f"{nvcf_url}/v2/nvcf"
        self.asset_handler = NvidiaAssetClient(self.token_manager,
                                               self.endpoint)
        self.client_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0,
                                               enable_cleanup_closed=True))

    async def close(self):
        await self.client_session.close()

    # Invoke a function
    async def nvidia_post_call(
        self,
        token: str,
        nvidia_request: NvidiaRequest,
        task_id: str,
    ) -> Tuple[ClientResponse, List[str]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "NVCF-POLL-SECONDS": NVCF_POLL_SECONDS,
        }

        nvidia_function = nvidia_request.function_id
        nvcf_function_inputs = nvidia_request.parameters
        data = {
            "inputs": [
                process_parameter(name, parameter)
                for name, parameter in nvcf_function_inputs.items()
                if parameter is not None
                   and (
                           not isinstance(parameter, NvidiaRequestParameter)
                           or parameter.value is not None
                   )
            ],
            "outputs": [
                {
                    "name": nvidia_request.image_output_name,
                    "datatype": "BYTES",
                    "shape": [1],
                }
            ],
        }

        if nvidia_request.profile_output_name:
            data["outputs"].append(
                {
                    "name": nvidia_request.profile_output_name,
                    "datatype": "BYTES",
                    "shape": [1],
                }
            )

        assets, data, headers = await self.asset_handler.handle_assets(
            self.client_session, nvidia_request, token, data, headers
        )

        try:
            payload = json.dumps(data)
            post_url = f"{self.endpoint}/pexec/functions/{nvidia_function}"
            logger.info(f"Sending {task_id} to {post_url} with payload: {payload}")
            async with self.client_session.post(
                post_url,
                headers=headers,
                data=payload,
            ) as response:
                if not is_response_status_valid(response) and response.status != 302:
                    exception_reason = await response.text()
                    check_custom_exception_reasons(
                        nvidia_request, task_id, response.status, exception_reason
                    )
                    raise NvidiaPostClientException(
                        nvidia_request,
                        task_id,
                        post_url,
                        response.status,
                        exception_reason,
                        payload,
                    )

                await response.read()  # Load body as to not need the connection to stay alive

                return response, assets
        except Exception as e:
            # If we fail, handle cleaning assets before returning
            await self.asset_handler.cleanup_assets(self.client_session, assets, token)
            raise e

    async def get_request_status_by_id(
        self,
        req_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> ClientResponse:
        if req_id is None:
            raise InvalidNvidiaPollParamsException(
                "Received 202 but no request id header was present"
            )
        token = await self.token_manager.fetch_token_if_required(self.client_session, token)
        headers = {"Authorization": f"Bearer {token}",
                   "NVCF-POLL-SECONDS": NVCF_POLL_SECONDS}
        get_url = f"{self.endpoint}/pexec/status/{req_id}"

        async with self.client_session.get(get_url, headers=headers) as response:
            await response.read()  # may be a 302, which has no json body

            return response

    async def handle_response(
        self,
        response: ClientResponse,
        nvidia_request: NvidiaRequest,
        task_id: str,
        last_request_time: float,
        token: str,
    ) -> Tuple[bytes, List[Any]]:
        num_requests: int = 0  # The number of times we have polled for the request
        while num_requests <= config.NVCF_MAX_POLLING_ATTEMPTS:
            if response.status == 200 or response.status == 302:
                req_id = response.headers.get("NVCF-REQID")
                logger.info(
                    f"task_id: {task_id} req_id: {req_id} fulfilled in {num_requests} polls"
                )
                # if there is a responseReference, we need to get the image from the URL
                return await handle_fulfilled_response(
                        self.client_session, response, nvidia_request, task_id,
                        req_id
                )

            elif response.status == 202:
                await response.json()  # drain body for connection reuse
                req_id = response.headers.get("NVCF-REQID")
                if num_requests >= config.NVCF_MAX_POLLING_ATTEMPTS:
                    raise NvidiaPollTimeoutException(nvidia_request, task_id, req_id)
                await explicitly_sleep_for_minimum_polling_interval(last_request_time)
                logger.info(f"task_id: {task_id} req_id: {req_id} still polling")
                # poll get_req_by_id until status is fulfilled
                response = await self.get_request_status_by_id(
                    req_id, token
                )
                num_requests += 1
                last_request_time = time.time()
            else:
                exception_reason = await response.text()
                check_custom_exception_reasons(
                    nvidia_request, task_id, response.status, exception_reason
                )
                raise NvidiaPollException(
                    nvidia_request, task_id, "", response.status, exception_reason
                )

    async def generate_image(
        self, nvidia_client_request: NvidiaRequest, task_id: str
    ) -> Optional[Tuple[bytes, List[Any]]]:
        # Get an auth token as Before we make a request, we need to make sure we have a valid auth token
        token = await self.token_manager.fetch_token_if_required(self.client_session)
        start_time_post = time.time()
        logger.info(f"Sending {task_id} to {self.endpoint} at {start_time_post}")

        # Invoke a function
        invoke_res, assets = await self.nvidia_post_call(
            token, nvidia_client_request, task_id
        )
        poll_start_time = time.time()
        response = None
        reason_for_failure = None
        try:
            response = await self.handle_response(
                invoke_res,
                nvidia_client_request,
                task_id,
                poll_start_time,
                token,
            )
            time_image_generation = time.time() - start_time_post
            logger.info(
                f"Image generation for {task_id} successful in {time_image_generation} seconds"
            )
        except Exception as e:
            reason_for_failure = str(e)
            logger.error(
                f"Nvidia call failed for {task_id}: {nvidia_client_request} due to {reason_for_failure}",
                exc_info=True,
            )

        await self.asset_handler.cleanup_assets(self.client_session, assets, token)

        return response, reason_for_failure
