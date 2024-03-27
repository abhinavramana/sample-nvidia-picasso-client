from io import BytesIO

from wombo_utilities import get_logger_for_file
from wombo_utilities.interface.nvidia.nvidia_request_models import (
    GuidanceNvidiaClientRequest,
)

from wombo.config import NVCF_UPSCALER_FUNCTION_ID
from wombo.nvidia import MIME_JPEG_CONTENT_TYPE
from wombo.nvidia.client.nvidia_request import NvidiaRequest, asset_from_bytes, NvidiaRequestParameter
from wombo.nvidia.nvidia_service import process_text_to_image, handle_custom_request

logger = get_logger_for_file(__name__)


async def multi_client_request(request: GuidanceNvidiaClientRequest) -> BytesIO:
    original_requested_width = request.width
    original_requested_height = request.height
    picasso_request_text2img = process_text_to_image(request)
    metric_attributes = {
        "task": "txt2img",
        "model": request.model,
    }
    generated_image_small = await handle_custom_request(
        picasso_request_text2img, request.task_id, metric_attributes
    )

    async def base_asset():
        return asset_from_bytes(
            generated_image_small,
            MIME_JPEG_CONTENT_TYPE,
        )

    picasso_request_upscale = NvidiaRequest(
        function_id=NVCF_UPSCALER_FUNCTION_ID,
        parameters={
            "desired_width": NvidiaRequestParameter(original_requested_width, "UINT16"),
            "desired_height": NvidiaRequestParameter(original_requested_height, "UINT16"),
        },
        assets={
            "original_image": base_asset,
        },
    )
    logger.info(f"Upscaling image with request: {picasso_request_upscale} for task {request.task_id}")
    upscaled_image = await handle_custom_request(
        picasso_request_upscale, request.task_id, None
    )
    return upscaled_image
