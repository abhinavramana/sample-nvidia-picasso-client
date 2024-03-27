from enum import Enum
from typing import Optional

from pydantic import Field, BaseModel

from sample_client_api.nvidia_request_models import ImageInput, DiffusionStyleParams


class BaseNvidiaClientRequest(BaseModel):
    model: Optional[str] = None
    task_id: str
    s3_output_bucket: Optional[str] = None
    s3_output_key: Optional[str] = None


class NvidiaClientRequest(BaseNvidiaClientRequest):
    prompt: str
    width: int
    height: int
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


class NoDimensionNvidiaClientRequest(BaseNvidiaClientRequest):
    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


class NvidiaScheduler(Enum):
    PNDM = "PNDM"
    LMSD = "LMSD"
    DPM = "DPM"
    DDIM = "DDIM"
    EulerA = "EulerA"


class GuidanceNvidiaClientRequest(NvidiaClientRequest):
    guidance: Optional[float] = Field(
        ge=0.0, le=40.0, default=None
    )  # txt2image, image2image


class NoDimensionGuidanceNvidiaClientRequest(NoDimensionNvidiaClientRequest):
    guidance: Optional[float] = Field(
        ge=0.0, le=40.0, default=None
    )  # txt2image, image2image


class StepsNvidiaClientRequest(NvidiaClientRequest):
    steps: Optional[int] = Field(ge=0, le=100, default=None)


class NoDimensionStepsNvidiaClientRequest(NoDimensionNvidiaClientRequest):
    steps: Optional[int] = Field(ge=0, le=100, default=None)


class SDXLNvidiaClientRequest(GuidanceNvidiaClientRequest):
    batch_size: Optional[int] = None
    base_steps: Optional[int] = 30
    refiner_steps: Optional[int] = 10
    scheduler: Optional[NvidiaScheduler] = None


class ImageNvidiaClientRequest(NvidiaClientRequest):
    image: ImageInput  # instructpix, image2image


class ImageToImageNvidiaClientRequest(
    GuidanceNvidiaClientRequest, ImageNvidiaClientRequest
):
    pass


class TextToVideoNvidiaClientRequest(GuidanceNvidiaClientRequest):
    num_frames: int = Field(default=30, ge=1, le=65535)


class InpaintNvidiaClientRequest(NvidiaClientRequest):
    input_image: ImageInput
    input_mask: ImageInput


class InstructNvidiaClientRequest(ImageNvidiaClientRequest):
    image_guidance: Optional[float] = Field(default=1.5, ge=1.0, le=32767.0)


class FaceswapNvidiaClientRequest(
    NoDimensionGuidanceNvidiaClientRequest, NoDimensionStepsNvidiaClientRequest
):
    source_image: ImageInput
    target_image: ImageInput
    input_image_strength: float = 0.3
    do_ip_adapter: bool = True
    ip_scale: Optional[float] = 0.3
    face_index: Optional[int] = 0
    allow_nsfw: bool = False
    img_nsfw_threshold: float = 0.9


class DiffusionNvidiaClientRequest(BaseNvidiaClientRequest):
    user_prompt: str
    desired_final_width: int
    desired_final_height: int

    style_params: DiffusionStyleParams

    allow_nsfw: bool = False
    seed: Optional[int] = None

    input_image: Optional[ImageInput] = None
    mask_image: Optional[ImageInput] = None


class FaceswapIpNvidiaClientRequest(
    GuidanceNvidiaClientRequest, StepsNvidiaClientRequest
):
    checkpoint: str

    enable_high_resolution_resample: bool = False
    high_resolution_resample_scale_factor: float = 1.5
    high_resolution_resample_steps: int = 20
    high_resolution_resample_strength: float = 0.5
    high_resolution_resample_ip_scale: float = 0.5
    high_resolution_resample_guidance_scale: float = 1.0

    enable_gfpgan: bool = True

    ip_image: ImageInput
    target_image: Optional[ImageInput] = None
    allow_nsfw: bool = False
    img_nsfw_threshold: float = 0.9


class AvatarNvidiaClientRequest(GuidanceNvidiaClientRequest, StepsNvidiaClientRequest):
    source_image: ImageInput
