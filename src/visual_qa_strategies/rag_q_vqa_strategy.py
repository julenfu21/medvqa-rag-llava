from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.utils.data_definitions import ArgumentSpec, ModelAnswerResult
from src.utils.enums import VQAStrategyType
from src.utils.logger import LoggerManager
from src.utils.prompts.prompts_helpers import log_conversation_messages
from src.utils.text_splitters.base_splitter import BaseSplitter
from src.visual_qa_strategies.base_rag_strategy import BaseRagVQAStrategy


class RagQVQAStrategy(BaseRagVQAStrategy):

    @property
    def strategy_type(self) -> VQAStrategyType:
        return VQAStrategyType.RAG_Q

    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        base64_image: Optional[str],
        logger_manager: Optional[LoggerManager],
        **kwargs: dict[str, Any]
    ) -> ModelAnswerResult:
        super()._validate_arguments(
            required_arguments=[
                ArgumentSpec(
                    name="doc_splitter",
                    expected_type=BaseSplitter,
                    is_optional=True
                )
            ],
            **kwargs
        )

        formatted_answers = " ".join(
            [f"{letter} - {answer}" for letter, answer in possible_answers.items()]
        )
        question_with_possible_answers = f"{question} {formatted_answers}"

        document_retrieval_result = self._retriever.get_formatted_relevant_documents(
            query=question,
            doc_splitter=kwargs.get("doc_splitter")
        )
        output = model.invoke({
            "question": question_with_possible_answers,
            "image": base64_image,
            "relevant_docs": document_retrieval_result.formatted_docs,
            "logger_manager": logger_manager
        })

        model_answer = output.strip()
        if logger_manager:
            log_conversation_messages(
                logger_manager=logger_manager,
                messages=[AIMessage(content=model_answer)]
            )
        return ModelAnswerResult(
            answer=model_answer,
            original_relevant_documents=document_retrieval_result.relevant_documents,
            shortened_relevant_documents=document_retrieval_result.split_documents
        )
