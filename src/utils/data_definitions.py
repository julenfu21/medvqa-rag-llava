from dataclasses import dataclass, field
from typing import Any, Type

from langchain_core.documents import Document


@dataclass
class ModelAnswerResult:
    answer: str
    relevant_documents: list[Document] = field(default_factory=list)


@dataclass
class ArgumentSpec:
    name: str
    expected_type: Type
    default_value: Any = None
    description: str = ""
