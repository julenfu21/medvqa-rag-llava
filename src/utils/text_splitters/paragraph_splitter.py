from langchain_core.documents import Document

from src.utils.enums import DocumentSplitterType
from src.utils.text_splitters.base_splitter import BaseSplitter


class ParagraphSplitter(BaseSplitter):

    @property
    def document_splitter_type(self) -> DocumentSplitterType:
        return DocumentSplitterType.PARAGRAPH_SPLITTER

    def split_documents(self, documents: list[Document]) -> list[str]:
        shortened_documents = []

        for document in documents:
            page_content = document.page_content
            split_paragraphs = [par.strip() for par in page_content.split('\n\n')]
            split_paragraphs = self._add_title_if_needed(split_document=split_paragraphs)
            if self._add_title:
                self._token_count += 1

            shortened_paragraphs = split_paragraphs[:self._token_count]
            shortened_text = "\n\n".join(shortened_paragraphs)
            shortened_documents.append(shortened_text)

        return shortened_documents
