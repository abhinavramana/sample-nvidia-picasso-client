import asyncio
import base64
import io
from typing import Tuple, Optional, TypeVar, Callable, Dict
import json

import PIL
import aioboto3
import numpy
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from starlette import status
from wombo_utilities import get_logger_for_file
from wombo_utilities.interface.art_inference.text_to_image_tasks import ImageInput
from wombo_utilities.interface.nvidia.nvidia_request_models import (
    NvidiaClientRequest,
    InstructNvidiaClientRequest,
    InpaintNvidiaClientRequest,
    ImageToImageNvidiaClientRequest,
    GuidanceNvidiaClientRequest,
    FaceswapNvidiaClientRequest,
    FaceswapIpNvidiaClientRequest,
    AvatarNvidiaClientRequest,
    DiffusionNvidiaClientRequest,
    BaseNvidiaClientRequest,
)

from wombo import config
from wombo.api.network_models import (
    NvidiaOutput,
)
from wombo.bootup.nvidia_objects import IMMUTABLE_BOOTUP_MANAGER
from wombo.config import (
    NVCF_INPAINT_FUNCTION_ID,
    NVIDIA_S3_BUCKET,
    NVCF_INSTRUCT_FUNCTION_ID,
    NVCF_FACESWAP_FUNCTION_ID,
    NVCF_FACESWAP_IP_FUNCTION_ID,
    NVCF_AVATAR_FUNCTION_ID,
)
from wombo.model_constants import SD_XL_0_9
from wombo.nvidia.client.nvidia_request import (
    NvidiaRequest,
    STYLES_TO_NVIDIA_FUNCTIONS,
    STYLES_TO_IMG2IMG_NVIDIA_FUNCTIONS,
    NvidiaRequestParameter,
    AssetLoader,
    asset_from_image,
)
from wombo.synth.synth_defaults import (
    SDXL_BASE_STEPS,
    T2I_SCHEDULER_STEPS,
    I2I_SCHEDULER_STEPS,
    INSTRUCT_SCHEDULER_STEPS,
    FACESWAP_BASE_STEPS,
    SD_CFG_SCALE_DEFAULT,
    INSTRUCT_IMAGE_CFG_MIN,
    INSTRUCT_IMAGE_CFG_MAX,
)
from wombo.synth.synth_spec_resolution_scaling import compute_base_dimensions

log = get_logger_for_file(__name__)

session = aioboto3.Session()

T = TypeVar("T", bound=BaseNvidiaClientRequest)


async def __upload_to_s3(fileobj: io.BytesIO, request: T) -> str:
    output_bucket = request.s3_output_bucket or NVIDIA_S3_BUCKET
    output_key = request.s3_output_key or f"{request.task_id}.jpeg"
    async with session.client("s3") as s3_client:
        await s3_client.upload_fileobj(fileobj, output_bucket, output_key)
        s3_uri = f"s3://{output_bucket}/{output_key}"
        log.info(f"Uploaded to {s3_uri}")

    return s3_uri


def __construct_asset(
        target: Optional[ImageInput],
        width: Optional[int] = None,
        height: Optional[int] = None,
) -> Optional[AssetLoader]:
    if target is None:
        return None

    async def load():
        log.info(
            f"Loading image from {target.image_bucket}/{target.image_key}{f',resizing to {width}x{height}' if width and height else ''} "
        )

        loop = asyncio.get_running_loop()
        data = io.BytesIO()
        async with session.client("s3") as s3_client:
            await s3_client.download_fileobj(
                target.image_bucket, target.image_key, data
            )

        if width and height:
            image = await loop.run_in_executor(None, lambda: PIL.Image.open(data))
            image_format = image.format

            image = await loop.run_in_executor(
                None, lambda: image.resize((width, height))
            )
        else:
            image = await loop.run_in_executor(None, lambda: PIL.Image.open(data))
            image_format = image.format

        return asset_from_image(image, image_format)

    return load


def __request_resolution(
        request: NvidiaClientRequest,
) -> Tuple[NvidiaRequestParameter, NvidiaRequestParameter]:
    width, height = compute_base_dimensions(
        should_lower_resolution_drastically=config.DO_V1_LOWER_RES,
        is_sd_xl=False,
        final_model=request.model,
        final_width=request.width,
        final_height=request.height,
    )

    return NvidiaRequestParameter(width, "UINT16"), NvidiaRequestParameter(
        height, "UINT16"
    )


def __request_model(request: T) -> str:
    return request.model or config.DEFAULT_STYLE_MODEL


def __request_guidance(request: GuidanceNvidiaClientRequest):
    return request.guidance or SD_CFG_SCALE_DEFAULT


def __instruct_guidance():
    return np.random.uniform(low=INSTRUCT_IMAGE_CFG_MIN, high=INSTRUCT_IMAGE_CFG_MAX)


def __get_seed(seed: Optional[int]) -> NvidiaRequestParameter:
    if seed is None:
        seed = numpy.random.randint(0, int(1e9))

    return NvidiaRequestParameter(seed, "UINT32")


async def handle_request(
        client_request: T, request_factory: Callable[[T], NvidiaRequest]
):
    request = request_factory(client_request)
    metric_attributes = {
        "task": request_factory.__name__,
        "model": client_request.model,
    }

    result = await IMMUTABLE_BOOTUP_MANAGER.nvidia_task_handler.handle_nvidia_task(
        request, client_request.task_id, metric_attributes
    )

    (file, outputs) = result

    if len(outputs):
        profile = json.loads(outputs[0])
    else:
        profile = {}

    return NvidiaOutput(
        output=await __upload_to_s3(io.BytesIO(file), client_request), profile=profile
    )


async def handle_custom_request(
        request: NvidiaRequest,
        task_id: str,
        metric_attributes: Optional[Dict[str, str]] = None,
):
    result = await IMMUTABLE_BOOTUP_MANAGER.nvidia_task_handler.handle_nvidia_task(
        request, task_id, metric_attributes
    )

    (file, _) = result

    return io.BytesIO(file)


def process_text_to_image(
        request: GuidanceNvidiaClientRequest,
) -> NvidiaRequest:
    width, height = __request_resolution(request)

    return NvidiaRequest(
        function_id=STYLES_TO_NVIDIA_FUNCTIONS[__request_model(request)],
        parameters={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": width,
            "height": height,
            "guidance": __request_guidance(request),
            "steps": NvidiaRequestParameter(T2I_SCHEDULER_STEPS, "UINT16"),
            "seed": __get_seed(request.seed),
        },
    )


def process_image_to_image(
        request: ImageToImageNvidiaClientRequest,
) -> NvidiaRequest:
    steps = SDXL_BASE_STEPS if request.model == SD_XL_0_9 else I2I_SCHEDULER_STEPS
    width, height = __request_resolution(request)
    return NvidiaRequest(
        function_id=STYLES_TO_IMG2IMG_NVIDIA_FUNCTIONS[__request_model(request)],
        parameters={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": width,
            "height": height,
            "strength": request.image.weight,
            "guidance": __request_guidance(request),
            "steps": NvidiaRequestParameter(steps, "UINT16"),
            "seed": __get_seed(request.seed),
        },
        assets={
            "image": __construct_asset(request.image, width.value, height.value),
        },
    )


def process_inpaint(request: InpaintNvidiaClientRequest) -> NvidiaRequest:
    width, height = __request_resolution(request)

    return NvidiaRequest(
        function_id=NVCF_INPAINT_FUNCTION_ID,
        parameters={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": width,
            "height": height,
            "strength": 1.0,
            "guidance": __instruct_guidance(),
            "steps": NvidiaRequestParameter(INSTRUCT_SCHEDULER_STEPS, "UINT16"),
            "seed": __get_seed(request.seed),
        },
        assets={
            "input_image": __construct_asset(
                request.input_image, width.value, height.value
            ),
            "input_mask": __construct_asset(
                request.input_mask, width.value, height.value
            ),  # TODO Need to invert
        },
    )


def process_instruct(request: InstructNvidiaClientRequest) -> NvidiaRequest:
    width, height = __request_resolution(request)

    return NvidiaRequest(
        function_id=NVCF_INSTRUCT_FUNCTION_ID,
        parameters={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": width,
            "height": height,
            "guidance": __instruct_guidance(),
            "steps": NvidiaRequestParameter(INSTRUCT_SCHEDULER_STEPS, "UINT16"),
            "seed": __get_seed(request.seed),
        },
        assets={
            "image": __construct_asset(request.image, width.value, height.value),
        },
    )


def process_faceswap(request: FaceswapNvidiaClientRequest) -> NvidiaRequest:
    params = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "steps": NvidiaRequestParameter(request.steps or FACESWAP_BASE_STEPS, "UINT16"),
        "seed": __get_seed(request.seed),
        "guidance": request.guidance,
        "input_image_strength": request.input_image_strength,
        "ip_scale": request.ip_scale,
    }
    if config.SEND_NSFW_PARAMS:
        params["allow_nsfw"] = request.allow_nsfw
        params["img_nsfw_threshold"] = request.img_nsfw_threshold
    if config.DO_FACE_INDEX:
        params["face_index"] = NvidiaRequestParameter(request.face_index, "UINT16")
    if config.DO_IP_ADAPTER:
        params["do_ip_adapter"] = request.do_ip_adapter
    return NvidiaRequest(
        function_id=NVCF_FACESWAP_FUNCTION_ID,
        parameters=params,
        assets={
            "source_image": __construct_asset(request.source_image),
            "target_image": __construct_asset(request.target_image),
        },
    )


def process_faceswap_ip_adapter(
        request: FaceswapIpNvidiaClientRequest,
) -> NvidiaRequest:
    parameters = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "width": request.width,
        "height": request.height,
        "steps": NvidiaRequestParameter(request.steps, "UINT16"),
        "seed": __get_seed(request.seed),
        "guidance": request.guidance,
        "ip_scale": request.ip_image.weight,
        "input_image_strength": (
            request.target_image.weight if request.target_image else None
        ),
        "enable_high_resolution_resample": request.enable_high_resolution_resample,
        "high_resolution_resample_scale_factor": request.high_resolution_resample_scale_factor,
        "high_resolution_resample_steps": NvidiaRequestParameter(
            request.high_resolution_resample_steps, "UINT16"
        ),
        "high_resolution_resample_strength": request.high_resolution_resample_strength,
        "high_resolution_resample_ip_scale": request.high_resolution_resample_ip_scale,
        "high_resolution_resample_guidance_scale": request.high_resolution_resample_guidance_scale,
        "enable_gfpgan": request.enable_gfpgan,
        "checkpoint": request.checkpoint,
    }
    if config.SEND_NSFW_PARAMS_IP:
        parameters["allow_nsfw"] = request.allow_nsfw
        parameters["img_nsfw_threshold"] = request.img_nsfw_threshold
    return NvidiaRequest(
        function_id=NVCF_FACESWAP_IP_FUNCTION_ID,
        parameters=parameters,
        assets={
            "source_image": __construct_asset(request.ip_image),
            "target_image": __construct_asset(request.target_image),
        },
    )


def process_avatar(request: AvatarNvidiaClientRequest) -> NvidiaRequest:
    return NvidiaRequest(
        function_id=NVCF_AVATAR_FUNCTION_ID,
        parameters={
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": NvidiaRequestParameter(request.steps, "UINT16"),
            "seed": __get_seed(request.seed),
            "guidance": request.guidance,
            "ip_scale": request.source_image.weight,
            "width": request.width,
            "height": request.height,
        },
        assets={
            "source_image": __construct_asset(
                request.source_image, request.width, request.height
            ),
        },
    )


class NvidiaUpscalerRequest(BaseModel):
    original_image: ImageInput
    desired_width: int
    desired_height: int


def process_diffusion(
        request: DiffusionNvidiaClientRequest, function_id: str
) -> NvidiaRequest:
    return NvidiaRequest(
        function_id=function_id,
        parameters={
            "user_prompt": request.user_prompt,
            # TODO: temp fix of excluding model and allow_nsfw from style_params. Should remove
            # that after NVIDIA deploys our SDXL container
            "style_params": request.style_params.model_dump_json(
                exclude={"model", "allow_nsfw"}
            ),
            "seed": __get_seed(request.seed),
            "desired_final_width": request.desired_final_width,
            "desired_final_height": request.desired_final_height,
            "input_image_strength": (
                request.input_image.weight if request.input_image else None
            ),
        },
        assets={
            "input_image_path": __construct_asset(request.input_image),
            "mask_image_path": __construct_asset(request.mask_image),
        },
        profile_output_name="profile",
    )


def process_upscaler(request: NvidiaUpscalerRequest, function_id: str) -> NvidiaRequest:
    return NvidiaRequest(
        function_id=function_id,
        parameters={
            "desired_width": request.desired_width,
            "desired_height": request.desired_height,
        },
        assets={
            "original_image": __construct_asset(request.original_image),
        },
    )
