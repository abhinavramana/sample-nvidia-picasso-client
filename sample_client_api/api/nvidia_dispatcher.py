from sample_client_api.api.network_models import (
    NvidiaOutput,
)
from sample_client_api.config import (
    NVCF_SDXL_DIFFUSION_FUNCTION_ID,
    NVCF_UPSCALER_FUNCTION_ID,
)
from sample_client_api.custom_router import CustomAPIRouter
from sample_client_api.nvidia.nvidia_multi_client_request import multi_client_request
from sample_client_api.nvidia.nvidia_service import (
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
from sample_client_api.nvidia_request_models.final_models import (
    GuidanceNvidiaClientRequest,
    ImageToImageNvidiaClientRequest,
    InpaintNvidiaClientRequest,
    InstructNvidiaClientRequest,
    FaceswapNvidiaClientRequest,
    FaceswapIpNvidiaClientRequest,
    AvatarNvidiaClientRequest,
    DiffusionNvidiaClientRequest,
)

nvidia_dispatcher = CustomAPIRouter()


@nvidia_dispatcher.post("/txt2img", response_model=NvidiaOutput)
async def text_to_image_and_upscale(
    request: GuidanceNvidiaClientRequest,
) -> NvidiaOutput:
    output_bytes = await multi_client_request(request)
    result = await __upload_to_s3(output_bytes, request)
    return NvidiaOutput(output=result)


@nvidia_dispatcher.post("/img2img", response_model=NvidiaOutput)
async def image_to_image(
    request: ImageToImageNvidiaClientRequest
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
    request: FaceswapNvidiaClientRequest, 
) -> NvidiaOutput:
    return await handle_request(request, process_faceswap)


@nvidia_dispatcher.post("/faceswap_ip", response_model=NvidiaOutput)
async def faceswap_ip(
    request: FaceswapIpNvidiaClientRequest, 
) -> NvidiaOutput:
    return await handle_request(request, process_faceswap_ip_adapter)


@nvidia_dispatcher.post("/avatar", response_model=NvidiaOutput)
async def avatar(
    request: AvatarNvidiaClientRequest, 
) -> NvidiaOutput:
    return await handle_request(request, process_avatar)


@nvidia_dispatcher.post("/sdxl_diffusion", response_model=NvidiaOutput)
async def sdxl_diffusion(
    request: DiffusionNvidiaClientRequest, 
) -> NvidiaOutput:
    return await handle_request(
        request,
        lambda r: process_diffusion(
            r, NVCF_SDXL_DIFFUSION_FUNCTION_ID
        ),
    )


@nvidia_dispatcher.post("/upscaler", response_model=NvidiaOutput)
async def upscaler(
    request: NvidiaUpscalerRequest, 
) -> NvidiaOutput:
    return await handle_request(
        request, lambda r: process_upscaler(r, NVCF_UPSCALER_FUNCTION_ID)
    )

