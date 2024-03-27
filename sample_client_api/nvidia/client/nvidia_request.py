import io
import json
from io import BytesIO
from typing import Optional, Dict, Any, Callable, Awaitable

import PIL
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict
from sample_client_api.log_handling import get_logger_for_file

from sample_client_api import config
from sample_client_api.nvidia import MIME_JPEG_CONTENT_TYPE

log = get_logger_for_file(__name__)


def get_styles_to_nvidia_functions(
    function_id_name_json_str: str,
) -> Dict[str, str]:
    functions_dict = json.loads(function_id_name_json_str)
    log.info(f"Nvidia functions: {functions_dict}")
    return functions_dict


STYLES_TO_NVIDIA_FUNCTIONS = get_styles_to_nvidia_functions(
    config.DIFFUSION_STYLE_MODEL_TO_NVCF_FUNCTION
)

STYLES_TO_IMG2IMG_NVIDIA_FUNCTIONS = get_styles_to_nvidia_functions(
    config.IMG2IMG_STYLE_MODEL_TO_NVCF_FUNCTION
)

FACESWAP_FUNCTION_ID_SET = {
    config.NVCF_FACESWAP_FUNCTION_ID,
    config.NVCF_FACESWAP_IP_FUNCTION_ID,
}


class NvidiaRequestParameter:
    value: Any
    parameter_type: Optional[str]

    def __init__(self, value: Any, parameter_type: Optional[str] = None):
        self.value = value
        self.parameter_type = parameter_type

    def detect_type(self) -> str:
        if self.parameter_type is not None:
            return self.parameter_type

        value_type = type(self.value)

        if value_type is int:
            return "UINT32"
        elif value_type is float:
            return "FP32"
        elif value_type is bool:
            return "BOOL"
        else:
            return "BYTES"

    def __repr__(self) -> str:
        detected_type = (
            self.detect_type()
        )  # Utilize the existing method to get the type
        return f"NvidiaRequestParameter(value={self.value}, parameter_type={detected_type})"


class NvidiaRequestAsset(BaseModel):
    data: BytesIO
    content_type: str
    content_length: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __enter__(self):
        self.data.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data.__exit__(exc_type, exc_val, exc_tb)


AssetLoader = Callable[[], Awaitable[NvidiaRequestAsset]]


def asset_from_image(image: Image, image_format: str):
    image_data = io.BytesIO()
    image.save(image_data, format=image_format)
    return asset_from_bytes(
        image_data,
        PIL.Image.MIME.get(image_format, MIME_JPEG_CONTENT_TYPE),
    )


def asset_from_bytes(image_data: BytesIO, content_type: str):
    image_data.seek(0, io.SEEK_END)  # Go to the end of the file
    file_length = image_data.tell()  # Get the position of EOF
    image_data.seek(0)

    return NvidiaRequestAsset(
        data=image_data,
        content_type=content_type,
        content_length=file_length,
    )


class NvidiaRequest(BaseModel):
    function_id: str

    parameters: Dict[str, Any]
    assets: Dict[str, Optional[AssetLoader]] = {}

    image_output_name: str = "generated_image"
    profile_output_name: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
