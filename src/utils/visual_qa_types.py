from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class ModelAnswerResult:
    answer: str
    relevant_documents: list[Document] = field(default_factory=list)
