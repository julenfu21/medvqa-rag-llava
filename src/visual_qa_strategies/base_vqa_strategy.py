from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.enums import VQAStrategyType
from src.utils.string_formatting_helpers import prettify_strategy_name


class BaseVQAStrategy(ABC):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        formatted_strategy_name = prettify_strategy_name(strategy_name=self.strategy_type.value)
        print(f"- Loading {formatted_strategy_name} Strategy ...")
        self._init_strategy(*args, **kwargs)
        print(f"+ {formatted_strategy_name} Strategy Loaded.")

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
        base64_image: str
    ) -> str:
        pass
