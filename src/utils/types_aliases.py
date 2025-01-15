from typing import Union
from src.utils.enums import RagQPromptType, ZeroShotPromptType


PromptType = Union[ZeroShotPromptType, RagQPromptType]
