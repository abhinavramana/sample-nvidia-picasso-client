from io import BytesIO

from fastapi import Depends
from requests_toolbelt import MultipartEncoder
from starlette.responses import Response
from wombo_utilities.interface.nvidia.nvidia_request_models import (
    GuidanceNvidiaClientRequest,
    ImageToImageNvidiaClientRequest,
    InpaintNvidiaClientRequest,
    InstructNvidiaClientRequest,
    FaceswapNvidiaClientRequest,
    FaceswapIpNvidiaClientRequest,
    AvatarNvidiaClientRequest,
    DiffusionNvidiaClientRequest,
)

from wombo.api.network_models import (
    NvidiaOutput,
)
from wombo.bootup.auth import verify_token
from wombo.config import (
    NVCF_SDXL_DIFFUSION_FUNCTION_ID_CALLED_WOMBO_DIFFUSION,
    NVCF_UPSCALER_FUNCTION_ID,
)
from wombo.custom_router import WOMBOAPIRouter
from wombo.nvidia import MIME_JPEG_CONTENT_TYPE
from wombo.nvidia.nvidia_multi_client_request import multi_client_request
from wombo.nvidia.nvidia_service import (
    process_image_to_image,
    process_inpaint,
    process_instruct,
    handle_request,
    process_faceswap,
    process_faceswap_ip_adapter,
    process_avatar,
    process_diffusion,
    process_upscaler,
    NvidiaUpscalerRequest,
    __upload_to_s3,
)

nvidia_dispatcher = WOMBOAPIRouter()


@nvidia_dispatcher.post("/txt2img", response_model=NvidiaOutput)
async def text_to_image_and_upscale(
    request: GuidanceNvidiaClientRequest, token_verified: bool = Depends(verify_token)
) -> NvidiaOutput:
    output_bytes = await multi_client_request(request)
    result = await __upload_to_s3(output_bytes, request)
    return NvidiaOutput(output=result)


@nvidia_dispatcher.post("/img2img", response_model=NvidiaOutput)
async def image_to_image(
    request: ImageToImageNvidiaClientRequest,
    token_verified: bool = Depends(verify_token),
) -> NvidiaOutput:
    return await handle_request(request, process_image_to_image)


@nvidia_dispatcher.post("/inpaint", response_model=NvidiaOutput)
async def inpaint(request: InpaintNvidiaClientRequest) -> NvidiaOutput:
    return await handle_request(request, process_inpaint)


@nvidia_dispatcher.post("/instruct", response_model=NvidiaOutput)
async def instruct(request: InstructNvidiaClientRequest) -> NvidiaOutput:
    return await handle_request(request, process_instruct)


@nvidia_dispatcher.post("/faceswap", response_model=NvidiaOutput)
async def faceswap(
    request: FaceswapNvidiaClientRequest, token_verified: bool = Depends(verify_token)
) -> NvidiaOutput:
    return await handle_request(request, process_faceswap)


@nvidia_dispatcher.post("/faceswap_ip", response_model=NvidiaOutput)
async def faceswap_ip(
    request: FaceswapIpNvidiaClientRequest, token_verified: bool = Depends(verify_token)
) -> NvidiaOutput:
    return await handle_request(request, process_faceswap_ip_adapter)


@nvidia_dispatcher.post("/avatar", response_model=NvidiaOutput)
async def avatar(
    request: AvatarNvidiaClientRequest, token_verified: bool = Depends(verify_token)
) -> NvidiaOutput:
    return await handle_request(request, process_avatar)


@nvidia_dispatcher.post("/sdxl_diffusion", response_model=NvidiaOutput)
async def sdxl_diffusion(
    request: DiffusionNvidiaClientRequest, token_verified: bool = Depends(verify_token)
) -> NvidiaOutput:
    return await handle_request(
        request,
        lambda r: process_diffusion(
            r, NVCF_SDXL_DIFFUSION_FUNCTION_ID_CALLED_WOMBO_DIFFUSION
        ),
    )


@nvidia_dispatcher.post("/upscaler", response_model=NvidiaOutput)
async def upscaler(
    request: NvidiaUpscalerRequest, token_verified: bool = Depends(verify_token)
) -> NvidiaOutput:
    return await handle_request(
        request, lambda r: process_upscaler(r, NVCF_UPSCALER_FUNCTION_ID)
    )

