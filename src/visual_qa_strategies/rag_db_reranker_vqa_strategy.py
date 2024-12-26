from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.enums import VQAStrategyType
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


class RagDBRerankerVQAStrategy(BaseVQAStrategy):

    def _init_strategy(self) -> None:
        raise NotImplementedError


    @property
    def strategy_type(self):
        raise VQAStrategyType.RAG_DB_RERANKER


    def load_ollama_model(self, model_name: str) -> BaseChatModel:
        raise NotImplementedError


    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        base64_image: str
    ) -> str:
        raise NotImplementedError
