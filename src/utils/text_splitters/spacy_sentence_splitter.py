from typing import Union

import spacy
from langchain_core.documents import Document

from src.utils.enums import DocumentSplitterType
from src.utils.text_splitters.base_splitter import BaseSplitter


class SpacySentenceSplitter(BaseSplitter):

    def __init__(
        self,
        token_count: int,
        add_title: bool = False,
        model_name: str = "en_core_web_sm"
    ) -> None:
        super().__init__(token_count, add_title)
        self.__model_name = model_name
        self.__nlp = spacy.load(self.__model_name)

    @property
    def document_splitter_type(self) -> DocumentSplitterType:
        return DocumentSplitterType.SPACY_SENTENCE_SPLITTER

    def split_documents(self, documents: list[Document]) -> Union[list[str], list[Document]]:
        shortened_documents = []

        for document in documents:
            page_content = document.page_content
            split_paragraphs = [par.strip() for par in page_content.split('\n\n')]            
            split_paragraphs = self._add_title_if_needed(split_document=split_paragraphs)
            document_text = "\n\n".join(split_paragraphs)

            spacy_document = self.__nlp(document_text)
            split_sentences = [sentence.text.strip() for sentence in spacy_document.sents]
            shortened_sentences = split_sentences[:self._token_count]
            shortened_text = " ".join(shortened_sentences)
            shortened_documents.append(shortened_text)

        return shortened_documents
