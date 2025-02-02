from dataclasses import dataclass, field
from typing import Any, Callable, Type

from langchain_core.documents import Document

from src.utils.text_splitters.base_splitter import BaseSplitter


@dataclass
class ModelAnswerResult:
    answer: str
    original_relevant_documents: list[Document] = field(default_factory=list)
    shortened_relevant_documents: list[str] = field(default_factory=list)


@dataclass
class ArgumentSpec:
    name: str
    expected_type: Type
    value: Any = None
    is_optional: bool = False


@dataclass
class DocSplitOptions:
    chunk_size: int
    chunk_overlap: int
    short_docs_count: int


@dataclass
class EvaluationFolderHierarchy:
    second_level: str
    third_level: Callable[[BaseSplitter], str]


@dataclass
class ScriptArgument:
    name: str
    value: Any
    default_value: Any
    dependency_name: str
    dependency_value: Any
    valid_arg_condition: bool
    error_condition_message: str
