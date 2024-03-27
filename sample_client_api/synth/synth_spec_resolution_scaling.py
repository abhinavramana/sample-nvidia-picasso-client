from math import floor, ceil
from typing import Tuple

from wombo_utilities import get_logger_for_file

from wombo import config
from sample_client_api.model_constants import SD_V1_4, SD_V1_5, INSTRUCT
from sample_client_api.synth.synth_defaults import DIFFUSION_RES_DIVISOR

logger = get_logger_for_file(__name__)


# The config values should already be divisible by DIFFUSION_RES_DIVISOR, but
# we ensure that is the case here.
ART_MAX_DIM = (
    floor(config.ART_MAX_GENERATION_DIMENSION / DIFFUSION_RES_DIVISOR)
    * DIFFUSION_RES_DIVISOR
)
ART_MIN_DIM = (
    ceil(config.ART_MIN_GENERATION_DIMENSION / DIFFUSION_RES_DIVISOR)
    * DIFFUSION_RES_DIVISOR
)
SDXL_SCALE = 1.4
SD_V1_SCALE = 3.0
V1_MODELS = [SD_V1_4, SD_V1_5, INSTRUCT]
SD_SUPERRES_SCALE = 2


def handle_non_sdxl_resolution(base_width: int, base_height: int) -> Tuple[int, int]:
    # The compiled engines have max and min limits on the sizes they can generate. If both base_height and
    # base_width exceed the max dimension, the bigger of them needs to be adjusted to the max and then
    # the other one is adjusted to preserve the aspect ratio.

    if base_height >= base_width:
        # The cases of interest are that atleast one of them is greater than max (in which
        # case the height is capped to max and the width scaled since width is lesser) or
        # atleast one of them is lower than min (in which case the width is set to min and
        # height scaled since height is bigger).
        if base_height > ART_MAX_DIM:
            base_width = ceil(ART_MAX_DIM * base_width / base_height)
            base_height = ART_MAX_DIM
        if base_width < ART_MIN_DIM:
            base_height = floor(ART_MIN_DIM * base_height / base_width)
            base_width = ART_MIN_DIM
    else:
        # The cases of interest are one of them greater than max (in which case width, which is
        # larger is scaled to the max) or one of them less than min (in which case the height,
        # which is smaller, is set to the min).
        if base_width > ART_MAX_DIM:
            base_height = ceil(ART_MAX_DIM * base_height / base_width)
            base_width = ART_MAX_DIM
        if base_height < ART_MIN_DIM:
            base_width = floor(ART_MIN_DIM * base_width / base_height)
            base_height = ART_MIN_DIM

    return base_width, base_height


def aspect_ratio_specific_adjustments(
    base_width: int, base_height: int
) -> Tuple[int, int]:
    if base_height / base_width > config.ART_MAYBE_MAX_ASPECT_RATIO:
        # The height is too big and must be adjusted
        base_height = floor(base_width * config.ART_MAYBE_MAX_ASPECT_RATIO)
    elif base_width / base_height > config.ART_MAYBE_MAX_ASPECT_RATIO:
        # The width is too big and must be adjusted
        base_width = floor(base_height * config.ART_MAYBE_MAX_ASPECT_RATIO)
    # Otherwise, these dimensions are within acceptable AR bounds
    logger.info(
        f"After AR-specific adjustments, base_width={base_width}, base_height={base_height}"
    )
    return base_width, base_height


def find_base_dimensions(
    final_width: int, final_height: int, scale: float
) -> Tuple[int, int]:
    if final_height >= final_width:
        base_width = (
            round(final_width / scale / DIFFUSION_RES_DIVISOR) * DIFFUSION_RES_DIVISOR
        )
        base_height = (
            round(base_width * (final_height / final_width) / DIFFUSION_RES_DIVISOR)
            * DIFFUSION_RES_DIVISOR
        )
    else:
        base_height = (
            round(final_height / scale / DIFFUSION_RES_DIVISOR) * DIFFUSION_RES_DIVISOR
        )
        base_width = (
            round(base_height * (final_width / final_height) / DIFFUSION_RES_DIVISOR)
            * DIFFUSION_RES_DIVISOR
        )
    logger.info(
        f"Before hardware adjustments, base_width={base_width}, base_height={base_height}"
    )
    return base_width, base_height


def get_final_resolution_for_diffusion_res(
    is_sd_xl: bool, base_width: int, base_height: int
) -> Tuple[int, int]:
    # Now, ensure it is still divisible by DIFFUSION_RES_DIVISOR after these calculations and
    # clamp it to the upper/lower bounds
    base_width = base_width - base_width % DIFFUSION_RES_DIVISOR
    base_height = base_height - base_height % DIFFUSION_RES_DIVISOR
    if not is_sd_xl:  # only enforce min/max for non-SDXL
        base_width = min(ART_MAX_DIM, max(base_width, ART_MIN_DIM))
        base_height = min(ART_MAX_DIM, max(base_height, ART_MIN_DIM))
    logger.info(f"Final dimensions, base_width={base_width}, base_height={base_height}")
    return base_width, base_height


def choose_scaling(
    should_lower_resolution_drastically: bool,
    is_sd_xl: bool,
    final_model: str,
) -> float:
    scale = SD_SUPERRES_SCALE
    # We divide by 3 instead of 2 because instruct theoretically works better at lower res
    if should_lower_resolution_drastically and final_model in V1_MODELS:
        scale = SD_V1_SCALE
    if is_sd_xl:
        scale = SDXL_SCALE
    return scale


def compute_base_dimensions(
    should_lower_resolution_drastically: bool,
    is_sd_xl: bool,
    final_model: str,
    final_width: int,
    final_height: int,
) -> Tuple[int, int]:
    scale = choose_scaling(should_lower_resolution_drastically, is_sd_xl, final_model)
    base_width, base_height = find_base_dimensions(final_width, final_height, scale)

    # After we've done the above calculations, we might have to do 2 final adjustments:
    # 1) the aspect ratio may be too "elongated" in which case some hardware may not be able to generate it.
    if config.ART_MAYBE_MAX_ASPECT_RATIO:
        base_width, base_height = aspect_ratio_specific_adjustments(
            base_width, base_height
        )

    if not is_sd_xl:  # only enforce min/max for non-SDXL
        base_width, base_height = handle_non_sdxl_resolution(base_width, base_height)
    # 2) the dimensions may be too large for the compiled volta engines.
    # Some hardware may not have a limitation.
    # See if this one does.
    base_width, base_height = get_final_resolution_for_diffusion_res(
        is_sd_xl, base_width, base_height
    )
    return base_width, base_height
