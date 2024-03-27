import os

DDIM_ETA = 1.0
SD_CFG_SCALE_DEFAULT = 7.0
SD_CFG_SCALE_START = (
    float(os.getenv("SD_CFG_SCALE_START"))
    if "SD_CFG_SCALE_START" in os.environ
    else None
)
SD_CFG_SCALE_END = (
    float(os.getenv("SD_CFG_SCALE_END")) if "SD_CFG_SCALE_END" in os.environ else None
)
SD_VERSION = 1

SD_HEIGHT = 576
SD_WIDTH = 384
DIFFUSION_RES_DIVISOR = 64
SD_INSTRUCT_BASE = int(os.getenv("SD_INSTRUCT_BASE", 512))

N_STEPS_FIRST_PASS = 20
N_STEPS_SECOND_PASS = 20
N_STEPS_THIRD_PASS = 10
N_STEPS_LDM = N_STEPS_FIRST_PASS + N_STEPS_SECOND_PASS + N_STEPS_THIRD_PASS

DEFAULT_UNCOND_PROMPT = os.getenv(
    "DEFAULT_UNCOND_PROMPT",
    "typography, text, frame, cropped, signature, watermark, blurry, blur, sexy, nude, child, young, daughter, son, petite, furry, anthromorphic",
)

# Input Image Settings
INSTRUCT_IMAGE_CFG_MIN = float(os.getenv("INSTRUCT_IMAGE_CFG_MIN", 1.4))
INSTRUCT_IMAGE_CFG_MAX = float(os.getenv("INSTRUCT_IMAGE_CFG_MAX", 1.7))

# Scheduler Settings
T2I_SCHEDULER_STEPS = int(os.getenv("T2I_SCHEDULER_STEPS", 30))
I2I_SCHEDULER_STEPS = int(os.getenv("I2I_SCHEDULER_STEPS", 50))
INSTRUCT_SCHEDULER_STEPS = int(os.getenv("INSTRUCT_SCHEDULER_STEPS", 50))
SDXL_BASE_STEPS = int(os.getenv("SDXL_BASE_STEPS", 20))
FACESWAP_BASE_STEPS = int(os.getenv("FACESWAP_BASE_STEPS", 30))
