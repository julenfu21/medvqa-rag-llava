from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.data_definitions import ArgumentSpec, ModelAnswerResult
from src.utils.enums import VQAStrategyType
from src.utils.string_formatting_helpers import prettify_strategy_name
from src.utils.types_aliases import PromptType


class BaseVQAStrategy(ABC):

    def __init__(
        self,
        prompt_type: PromptType,
        *args: Any,
        **kwargs: dict[str, Any]
    ) -> None:
        self._prompt_type = prompt_type
        self._prompt_template = None

        formatted_strategy_name = prettify_strategy_name(strategy_name=self.strategy_type.value)
        print(f"- Loading {formatted_strategy_name} strategy ...")
        self._set_prompt_template()
        self._init_strategy(*args, **kwargs)
        print(f"+ {formatted_strategy_name} strategy loaded.")


    @abstractmethod
    def _init_strategy(
        self,
        *args: Any,
        **kwargs: dict[str, Any]
    ) -> None:
        pass


    def _validate_arguments(
        self,
        required_arguments: list[ArgumentSpec],
        **kwargs: dict[str, Any]
    ) -> None:
        if not required_arguments and kwargs:
            raise ValueError(
                f"The {self.strategy_type.value} does not accept any keyword argument, "
                f"but received: {list(kwargs.keys())}"
            )

        required_arguments_names = {argument.name for argument in required_arguments}
        extra_arguments = set(kwargs.keys()) - required_arguments_names
        if extra_arguments:
            raise ValueError(f"Unexpected keyword arguments: {extra_arguments}")

        for argument in required_arguments:
            self.__validate_argument(
                argument_name=argument.name,
                argument_value=argument.value if argument.value else kwargs.get(argument.name),
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
                f"got '{type(argument_value).__name__}'"
            )

    @property
    @abstractmethod
    def strategy_type(self) -> VQAStrategyType:
        pass

    @property
    def prompt_type(self) -> PromptType:
        return self._prompt_type

    @prompt_type.setter
    def prompt_type(self, prompt_type: PromptType) -> None:
        self._prompt_type = prompt_type
        self._set_prompt_template()

    @abstractmethod
    def _set_prompt_template(self) -> None:
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
    ) -> ModelAnswerResult:
        pass
