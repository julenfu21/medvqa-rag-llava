from pathlib import Path
from typing import Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.data_definitions import DocumentRetrievalResult
from src.utils.singleton_metaclass import SingletonMeta
from src.utils.text_splitters.base_splitter import BaseSplitter


class DocumentRetriever(metaclass=SingletonMeta):

    def __init__(
        self,
        index_dir: Path,
        index_name: str,
        embedding_model_name: str,
        relevant_docs_count: int
    ) -> None:
        print("\t- Loading Document Retriever ...")
        self.__retriever = self.__load_wikimed_retriever(
            index_dir, index_name, embedding_model_name, relevant_docs_count
        )
        print("\t+ Document Retriever Loaded.")

    def __load_wikimed_retriever(
        self,
        index_dir: Path,
        index_name: str,
        embedding_model_name: str,
        relevant_docs_count: int
    ) -> BaseRetriever:
        # Load embeddings
        print("\t\t- Loading Embeddings ...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            # model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        )
        print("\t\t+ Embeddings Loaded.")

        # Load FAISS index
        print("\t\t- Loading Index ...")
        index = FAISS.load_local(
            folder_path=index_dir,
            index_name=index_name,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("\t\t+ Index Loaded.")

        # Load retriever from index
        print("\t\t- Loading Retriever ...")
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": relevant_docs_count}
        )
        print("\t\t+ Retriever Loaded.")
        return retriever

    def get_formatted_relevant_documents(
        self,
        query: str,
        doc_splitter: BaseSplitter = None
    ) -> DocumentRetrievalResult:
        relevant_documents = self.__retriever.invoke(query)
        split_documents = (
            doc_splitter.split_documents(documents=relevant_documents)
            if doc_splitter else []
        )
        documents_to_format = split_documents or relevant_documents
        formatted_docs = self.__format_docs(docs=documents_to_format)

        return DocumentRetrievalResult(
            relevant_documents=relevant_documents,
            formatted_docs=formatted_docs,
            split_documents=split_documents
        )

    @staticmethod
    def __format_docs(docs: Union[list[str], list[Document]]) -> str:
        if isinstance(docs[0], str):
            return "\n\n".join(docs)

        if isinstance(docs[0], Document):
            return "\n\n".join(doc.page_content for doc in docs)

        raise TypeError(
            f"Expected elements within 'docs' to be of type '{str.__name__}' "
            f"or '{Document.__name__}', got '{type(docs[0].__name__)}'"
        )
