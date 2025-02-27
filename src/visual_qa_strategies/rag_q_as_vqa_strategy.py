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
        relevant_docs = ""
        question_relevant_documents = self._retriever.invoke(question)
        if should_apply_rag_to_question:
            relevant_docs += "RAG block for QUESTION:\n\n"
            if doc_splitter:
                split_documents = doc_splitter.split_documents(
                    documents=question_relevant_documents
                )
                relevant_docs += f"{super()._format_docs(docs=split_documents)}\n\n"
            else:
                relevant_docs += f"{super()._format_docs(docs=question_relevant_documents)}\n\n"

        relevant_docs += "RAG block for ANSWERS:\n\n"
        formatter_answer_rag_docs = []
        for letter, answer in possible_answers.items():
            answer_relevant_documents = self._retriever.invoke(answer)
            if doc_splitter:
                split_documents = doc_splitter.split_documents(documents=answer_relevant_documents)
                formatted_docs = super()._format_docs(docs=split_documents)
            else:
                formatted_docs = super()._format_docs(docs=answer_relevant_documents)
            formatter_answer_rag_docs.append(f"{letter} answer context:\n\n{formatted_docs}")
        relevant_docs += "\n\n".join(formatter_answer_rag_docs)

        output = model.invoke({
            "question": question_with_possible_answers,
            "image": base64_image,
            "relevant_docs": relevant_docs,
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
            original_relevant_documents=answer_relevant_documents,
            shortened_relevant_documents=split_documents if doc_splitter else []
        )
