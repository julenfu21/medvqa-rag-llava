from langchain_core.documents import Document

from src.utils.enums import DocumentSplitterType
from src.utils.text_splitters.base_splitter import BaseSplitter


class NoSplitter(BaseSplitter):

    @property
    def document_splitter_type(self) -> DocumentSplitterType:
        return DocumentSplitterType.NO_SPLITTER

    def split_documents(self, documents: list[Document]) -> list[str]:
        shortened_documents = []

        for document in documents:
            page_content = document.page_content
            split_paragraphs = [par.strip() for par in page_content.split('\n\n')]
            split_paragraphs = self._add_title_if_needed(split_document=split_paragraphs)
            shortened_documents.append("\n\n".join(split_paragraphs))

        return shortened_documents
