from src.utils.enums import DocumentSplitterType


DEFAULT_TOKEN_COUNT = {
    DocumentSplitterType.NO_SPLITTER: 1,
    DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER: 2,
    DocumentSplitterType.SPACY_SENTENCE_SPLITTER: 2,
    DocumentSplitterType.PARAGRAPH_SPLITTER: 1
}
DEFAULT_ADD_TITLE = False
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 0
