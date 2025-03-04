from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.utils.data_definitions import ArgumentSpec, ModelAnswerResult
from src.utils.document_retriever import DocumentRetriever
from src.utils.enums import RagQPromptType, VQAStrategyType
from src.utils.logger import LoggerManager
from src.utils.prompts.rag_q_prompts import RAG_Q_PROMPTS
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


class BaseRagVQAStrategy(BaseVQAStrategy, ABC):

    def _init_strategy(
        self,
        **kwargs: dict[str, Any]
    ) -> None:
        arguments = [
            ArgumentSpec(name="index_dir", expected_type=Path),
            ArgumentSpec(name="index_name", expected_type=str),
            ArgumentSpec(name="embedding_model_name", expected_type=str),
            ArgumentSpec(name="relevant_docs_count", expected_type=int)
        ]
        super()._validate_arguments(arguments, **kwargs)

        self.__relevant_docs_count = kwargs["relevant_docs_count"]
        self._retriever = DocumentRetriever(
            index_dir=kwargs["index_dir"],
            index_name=kwargs["index_name"],
            embedding_model_name=kwargs["embedding_model_name"],
            relevant_docs_count=self.__relevant_docs_count
        )

    @property
    def relevant_docs_count(self) -> int:
        return self.__relevant_docs_count

    def _set_prompt_template(self):
        super()._validate_arguments(
            required_arguments=[
                ArgumentSpec(
                    name="prompt_type", expected_type=RagQPromptType, value=self._prompt_type
                )
            ]
        )

        self._prompt_template = RAG_Q_PROMPTS[self._prompt_type]

    def load_ollama_model(self, model_name: str) -> BaseChatModel:
        llm = ChatOllama(model=model_name, temperature=0, num_predict=1)
        chain = self._prompt_template | llm | StrOutputParser()
        return chain

    @property
    @abstractmethod
    def strategy_type(self) -> VQAStrategyType:
        pass

    @abstractmethod
    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        base64_image: str,
        logger_manager: Optional[LoggerManager],
        **kwargs: dict[str, Any]
    ) -> ModelAnswerResult:
        pass
