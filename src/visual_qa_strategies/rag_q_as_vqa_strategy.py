from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.utils.data_definitions import ArgumentSpec, ModelAnswerResult
from src.utils.enums import VQAStrategyType
from src.utils.logger import LoggerManager
from src.utils.prompts.prompts_helpers import log_conversation_messages
from src.utils.text_splitters.base_splitter import BaseSplitter
from src.visual_qa_strategies.base_rag_strategy import BaseRagVQAStrategy


class RagQAsVQAStrategy(BaseRagVQAStrategy):

    @property
    def strategy_type(self) -> VQAStrategyType:
        return VQAStrategyType.RAG_Q_AS

    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        base64_image: str,
        logger_manager: Optional[LoggerManager],
        **kwargs: dict[str, Any]
    ) -> ModelAnswerResult:
        super()._validate_arguments(
            required_arguments=[
                ArgumentSpec(
                    name="doc_splitter",
                    expected_type=BaseSplitter,
                    is_optional=True
                ),
                ArgumentSpec(
                    name="should_apply_rag_to_question",
                    expected_type=bool,
                    is_optional=True
                )
            ],
            **kwargs
        )

        formatted_answers = " ".join(
            [f"{letter} - {answer}" for letter, answer in possible_answers.items()]
        )
        question_with_possible_answers = f"{question} {formatted_answers}"

        should_apply_rag_to_question = kwargs.get("should_apply_rag_to_question")
        doc_splitter = kwargs.get("doc_splitter")
        formatted_relevant_docs = ""
        if should_apply_rag_to_question:
            formatted_relevant_docs += "RAG block for QUESTION:\n\n"
            question_document_retrieval_result = self._retriever.get_formatted_relevant_documents(
                query=question,
                doc_splitter=doc_splitter
            )
            formatted_relevant_docs += f"{question_document_retrieval_result.formatted_docs}\n\n"

        formatted_relevant_docs += "RAG block for ANSWERS:\n\n"
        answers_document_retrieval_results = []
        formatted_answer_rag_docs = []
        for letter, answer in possible_answers.items():
            document_retrieval_result = self._retriever.get_formatted_relevant_documents(
                query=answer,
                doc_splitter=doc_splitter
            )
            formatted_answer_rag_docs.append(
                f"{letter} answer context:\n\n{document_retrieval_result.formatted_docs}"
            )
            answers_document_retrieval_results.append(document_retrieval_result)
        formatted_relevant_docs += "\n\n".join(formatted_answer_rag_docs)

        output = model.invoke({
            "question": question_with_possible_answers,
            "image": base64_image,
            "relevant_docs": formatted_relevant_docs,
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
            original_relevant_documents=[
                document
                for result in answers_document_retrieval_results
                for document in result.relevant_documents
            ],
            shortened_relevant_documents=[
                document
                for result in answers_document_retrieval_results
                for document in result.split_documents
            ]
        )
