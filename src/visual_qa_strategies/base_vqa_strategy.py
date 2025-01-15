from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.data_definitions import ArgumentSpec, ModelAnswerResult
from src.utils.enums import VQAStrategyType
from src.utils.string_formatting_helpers import prettify_strategy_name
from src.utils.types import PromptType


class BaseVQAStrategy(ABC):

    def __init__(
        self,
        prompt_type: PromptType,
        *args: Any,
        **kwargs: dict[str, Any]
    ) -> None:
        self._prompt_type = prompt_type

        formatted_strategy_name = prettify_strategy_name(strategy_name=self.strategy_type.value)
        print(f"- Loading {formatted_strategy_name} strategy ...")
        self._init_strategy(self._prompt_type, *args, **kwargs)
        print(f"+ {formatted_strategy_name} strategy loaded.")


    @abstractmethod
    def _init_strategy(
        self,
        prompt_type: PromptType,
        *args: Any,
        **kwargs: dict[str, Any]
    ) -> None:
        pass


    def _validate_kwargs(
        self,
        expected_arguments: list[ArgumentSpec],
        **kwargs: dict[str, Any]
    ) -> None:
        for argument in expected_arguments:
            self.__validate_argument(
                argument_name=argument.name,
                argument_value=kwargs.get(argument.name),
                expected_type=argument.expected_type
            )


    def __validate_argument(
        self,
        argument_name: str,
        argument_value: Any,
        expected_type: Any
    ) -> None:
        if argument_value is None:
            raise ValueError(f"Missing required argument: '{argument_name}'")
        if not isinstance(argument_value, expected_type):
            raise TypeError(
                f"Expected '{argument_name}' to be of type '{expected_type.__name__}', "
                f"got {type(argument_value).__name__}"
            )

    @property
    @abstractmethod
    def strategy_type(self) -> VQAStrategyType:
        pass

    @property
    def prompt_type(self) -> PromptType:
        return self._prompt_type


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
    ) -> ModelAnswerResult:
        pass
