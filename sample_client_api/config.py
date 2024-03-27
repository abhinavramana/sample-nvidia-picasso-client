import os

from wombo_utilities.interface import get_boolean_from_os

API_DOCS_ENDPOINT = os.getenv(
    "API_DOCS_ENDPOINT", "/docs"
)

NVIDIA_S3_BUCKET = os.getenv("NVIDIA_S3_BUCKET", "nvidia-generated-images")

DO_V1_LOWER_RES = get_boolean_from_os("DO_V1_LOWER_RES", default_value=True)

ART_MAX_GENERATION_DIMENSION = int(os.getenv("ART_MAX_GENERATION_DIMENSION", 1024))
ART_MIN_GENERATION_DIMENSION = int(os.getenv("ART_MIN_GENERATION_DIMENSION", 512))
# A max aspect ratio may or may not exist based on the hardware being used. A10G (G5) cannot generate
# aspect ratios (height/width or width/height) being > 3.5 but A100 can on the volta compilations
ART_MAYBE_MAX_ASPECT_RATIO = (
    float(os.getenv("ART_MAYBE_MAX_ASPECT_RATIO"))
    if os.getenv("ART_MAYBE_MAX_ASPECT_RATIO")
    else None
)

# NVIDIA API settings
NVCF_URL = os.getenv("NVCF_URL", "https://api.nvcf.nvidia.com")
NVCF_AUTH_URL = os.getenv(
    "NVCF_AUTH_URL",
    "SOME_AUTH_URL"
)
NVIDIA_CLIENT_SECRET = os.getenv("NVIDIA_CLIENT_SECRET", "nvidia_client_secret")
NVIDIA_USERNAME = os.getenv(
    "NVIDIA_USERNAME", "NVIDIA_USERNAME"
)
NVIDIA_PASSWORD_TO_RENEW_90_DAYS = os.getenv(
    "NVIDIA_WOMBO_PASSWORD_TO_RENEW_90_DAYS"
)
DEFAULT_STYLE_MODEL = os.getenv("DEFAULT_STYLE_MODEL", "stable_diffusion_1.5")
DIFFUSION_STYLE_MODEL_TO_NVCF_FUNCTION = os.getenv(
    "DIFFUSION_STYLE_MODEL_TO_NVCF_FUNCTION"
)
IMG2IMG_STYLE_MODEL_TO_NVCF_FUNCTION = os.getenv("IMG2IMG_STYLE_MODEL_TO_NVCF_FUNCTION")

NVCF_INPAINT_FUNCTION_ID = os.getenv("NVCF_INPAINT_FUNCTION_ID")
NVCF_INSTRUCT_FUNCTION_ID = os.getenv("NVCF_INSTRUCT_FUNCTION_ID")
NVCF_TXT2VID_FUNCTION_ID = os.getenv("NVCF_TXT2VID_FUNCTION_ID")
NVCF_FACESWAP_FUNCTION_ID = os.getenv("NVCF_FACESWAP_FUNCTION_ID")
NVCF_FACESWAP_IP_FUNCTION_ID = os.getenv("NVCF_FACESWAP_IP_FUNCTION_ID")
NVCF_AVATAR_FUNCTION_ID = os.getenv("NVCF_AVATAR_FUNCTION_ID")
NVCF_SDXL_DIFFUSION_FUNCTION_ID_CALLED_WOMBO_DIFFUSION = os.getenv(
    "NVCF_SDXL_DIFFUSION_FUNCTION_ID"
)
NVCF_UPSCALER_FUNCTION_ID = os.getenv("NVCF_UPSCALER_FUNCTION_ID")
NVCF_MAX_POLLING_ATTEMPTS = int(os.getenv("NVCF_MAX_POLLING_ATTEMPTS", 15))
NVCF_MIN_POLLING_INTERVAL = float(os.getenv("NVCF_MIN_POLLING_INTERVAL", 1.0))
NVCF_TOKEN_REFRESH_BUFFER_IN_SECONDS = int(
    os.getenv("NVCF_TOKEN_REFRESH_BUFFER_IN_SECONDS", 20)
)
DO_FACE_INDEX = get_boolean_from_os("DO_FACE_INDEX", False)
DO_IP_ADAPTER = get_boolean_from_os("DO_IP_ADAPTER", False)
SEND_NSFW_PARAMS = get_boolean_from_os("SEND_NSFW_PARAMS", False)
SEND_NSFW_PARAMS_IP = get_boolean_from_os("SEND_NSFW_PARAMS_IP", False)
