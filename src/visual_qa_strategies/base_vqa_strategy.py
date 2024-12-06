from abc import ABC, abstractmethod
from typing import Any

from PIL import Image
from langchain_core.language_models.chat_models import BaseChatModel

from utils.enums import VQAStrategyType


class BaseVQAStrategy(ABC):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        print(f"- Loading {self.strategy_type.value} Strategy ...")
        self._init_strategy(*args, **kwargs)
        print(f"+ {self.strategy_type.value} Strategy Loaded.")

    @abstractmethod
    def _init_strategy(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    @abstractmethod
    def strategy_type(self) -> VQAStrategyType:
        pass

    @abstractmethod
    def load_ollama_model(self, model_name: str) -> BaseChatModel:
        pass

    @abstractmethod
    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        image: Image.Image
    ) -> str:
        pass
