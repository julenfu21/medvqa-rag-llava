from abc import ABC, abstractmethod

from langchain_core.documents import Document

from src.utils.enums import DocumentSplitterType


class BaseSplitter(ABC):

    def __init__(self, token_count: int, add_title: bool = False) -> None:
        self._add_title = add_title
        self._token_count = token_count

    @property
    @abstractmethod
    def document_splitter_type(self) -> DocumentSplitterType:
        pass

    @property
    def add_title(self) -> bool:
        return self._add_title

    @property
    def token_count(self) -> int:
        return self._token_count

    @abstractmethod
    def split_documents(self, documents: list[Document]) -> list[str]:
        pass

    def _add_title_if_needed(
        self,
        split_document: list[str]
    ) -> list[str]:
        if self._add_title:
            return split_document

        return split_document[1:]
