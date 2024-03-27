from uuid import UUID
from enum import Enum, unique
from typing import Optional, List
from pydantic import field_validator, BaseModel
from datetime import datetime


class ImageInput(BaseModel):
    image_bucket: str
    image_key: str
    weight: float


class BaseTextToImageTask(BaseModel):
    id: UUID
    # Prompt settings
    user_prompt: str = ""
    allow_nsfw: bool = False
    image_nsfw_threshold: float = 0.5
    # Generation settings
    desired_final_height: int
    desired_final_width: int
    input_image_input: Optional[ImageInput] = None
    mask_image_input: Optional[ImageInput] = None

    output_s3_bucket: str
    # Images will be stored in the bucket in the format {output_s3_key_prefix}/{task_id}/{1,2,3,Final}.jpg
    output_s3_key_prefix: str
    # Each style may even specify very "internal" model parameters but not always.
    # We do not want the defaults for these fields to lie on the API side, to allow
    # ML folks to rapidly experiment and modify the defaults on their own cluster.
    # Thus, these values are "Optional" if not specified by the style config. These
    # are specified in the child classes.
    seed: Optional[int] = None
    created_at: datetime
    # This is used to put the base image uri to be accessible. Note that we are not using simple s3 key
    # because it can be used across projects with different s3 buckets. So keep it flexible
    base_generated_image_uri: Optional[str] = None

    @field_validator("watermark")
    @classmethod
    def enum_to_value(cls, v):
        # This ensures that the overall task is serializable when converted to dict by the API
        # worker to send down the task queue.
        return v.value if v is not None else None


class DiffusionStyleParams(BaseModel):
    prompt_template: str = "%"
    text_cfg: Optional[float] = None
    steps_override: Optional[int] = None
    uncond_prompt: Optional[str] = None
    # Internal SD height/width
    height: Optional[int] = None  # UNUSED, remove this param eventually
    width: Optional[int] = None  # UNUSED, remove this param eventually
    # Diffusion params
    nrow: int = 2
    sd_cfg_scale: Optional[float] = None
    sd_cfg_scale_start: Optional[float] = None  # UNUSED, remove this param eventually
    sd_cfg_scale_end: Optional[float] = None  # UNUSED, remove this param eventually
    model: Optional[str] = None
    # Scheduler params
    t2i_scheduler: Optional[str] = None
    t2i_scheduler_steps: Optional[int] = None
    i2i_scheduler: Optional[str] = None
    i2i_scheduler_steps: Optional[int] = None
    instruct_scheduler: Optional[str] = None
    instruct_scheduler_steps: Optional[int] = None
    ddim_eta: Optional[float] = None


class DiffusionTextToImageTask(BaseTextToImageTask):
    style_params: DiffusionStyleParams
