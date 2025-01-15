from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.data_definitions import ModelAnswerResult
from src.utils.enums import VQAStrategyType
from src.utils.types_aliases import PromptType
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


class RagImgVQAStrategy(BaseVQAStrategy):

    @property
    def strategy_type(self):
        raise VQAStrategyType.RAG_IMG


    def _init_strategy(
        self,
        prompt_type: PromptType,  # Change to specific PromptType
        *args: Any,
        **kwargs: dict[str, Any]
    ) -> None:
        raise NotImplementedError


    def load_ollama_model(self, model_name: str) -> BaseChatModel:
        raise NotImplementedError


    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        base64_image: str,
    ) -> ModelAnswerResult:
        raise NotImplementedError
