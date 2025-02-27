from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from src.utils.data_definitions import ArgumentSpec, ModelAnswerResult
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
        self._retriever = self.__load_wikimed_retriever(
            index_dir=kwargs["index_dir"],
            index_name=kwargs["index_name"],
            embedding_model_name=kwargs["embedding_model_name"],
            relevant_docs_count=self.__relevant_docs_count
        )

    def __load_wikimed_retriever(
        self,
        index_dir: Path,
        index_name: str,
        embedding_model_name: str,
        relevant_docs_count: int
    ) -> BaseRetriever:
        # Load embeddings
        print("\t- Loading Embeddings ...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            # model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        )
        print("\t+ Embeddings Loaded.")

        # Load FAISS index
        print("\t- Loading Index ...")
        index = FAISS.load_local(
            folder_path=index_dir,
            index_name=index_name,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("\t+ Index Loaded.")

        # Load retriever from index
        print("\t- Loading Retriever ...")
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": relevant_docs_count}
        )
        print("\t+ Retriever Loaded.")
        return retriever

    @property
    def relevant_docs_count(self) -> int:
        return self.__relevant_docs_count

    def _set_prompt_template(self):
        super()._validate_arguments(
            required_arguments=[
                ArgumentSpec(
                    name="prompt_type", expected_type=RagQPromptType, value=self._prompt_type
                    # name="prompt_type", expected_type=RagQAsPromptType, value=self._prompt_type
                )
            ]
        )

        self._prompt_template = RAG_Q_PROMPTS[self._prompt_type]
        # self._prompt_template = RAG_Q_AS_PROMPTS[self._prompt_type]


    def load_ollama_model(self, model_name: str) -> BaseChatModel:
        llm = ChatOllama(model=model_name, temperature=0, num_predict=1)
        chain = self._prompt_template | llm | StrOutputParser()
        return chain

    @staticmethod
    def _format_docs(docs: Union[list[str], list[Document]]) -> str:
        if isinstance(docs[0], str):
            return "\n\n".join(docs)

        if isinstance(docs[0], Document):
            return "\n\n".join(doc.page_content for doc in docs)

        raise TypeError(
            f"Expected elements within 'docs' to be of type '{str.__name__}' "
            f"or '{Document.__name__}', got '{type(docs[0].__name__)}'"
        )

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
