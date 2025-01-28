from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.enums import DocumentSplitterType
from src.utils.text_splitters.base_splitter import BaseSplitter


class RecursiveCharacterSplitter(BaseSplitter):

    def __init__(
        self,
        token_count: int,
        chunk_size: int,
        chunk_overlap: int,
        add_title: bool = False,
    ) -> None:
        super().__init__(token_count, add_title)
        self.__chunk_size = chunk_size
        self.__chunk_overlap = chunk_overlap
        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.__chunk_size,
            chunk_overlap=self.__chunk_overlap
        )

    @property
    def document_splitter_type(self) -> DocumentSplitterType:
        return DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER

    def split_documents(self, documents: list[Document]) -> list[str]:
        shortened_documents = []

        for document in documents:
            page_content = document.page_content
            split_paragraphs = [par.strip() for par in page_content.split('\n\n')]
            split_paragraphs = self._add_title_if_needed(split_document=split_paragraphs)
            document_text = "\n\n".join(split_paragraphs)

            if self._add_title:
                title_length = len(split_paragraphs[0])
                shortened_document = self.__text_splitter.split_text(
                    document_text[title_length + 2:]
                )[:self._token_count]
                shortened_document = f"{split_paragraphs[0]}\n\n{' '.join(shortened_document)}"
            else:
                shortened_document = self.__text_splitter.split_text(
                    document_text
                )[:self._token_count]
                shortened_document = " ".join(shortened_document)

            shortened_documents.append(shortened_document)

        return shortened_documents
