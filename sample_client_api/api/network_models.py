from typing import Dict, Any, Annotated, TypeAlias

from pydantic import BaseModel, Field

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 1344
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 7.0

MIN_SIZE = 512
MAX_SIZE = 1536
MAX_STEPS = 100

GenerationResolution = Annotated[int, Field(ge=MIN_SIZE, le=MAX_SIZE)]
Frames: TypeAlias = bytes


class NvidiaOutput(BaseModel):
    """
    Nvidia output model
    """

    output: str

    profile: Dict[str, Any] = {}


