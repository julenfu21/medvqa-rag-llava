from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from langchain_core.documents import Document

from src.utils.enums import DocumentSplitterType, VQAStrategyType
from src.utils.text_splitters.base_splitter import BaseSplitter
from src.utils.types_aliases import PromptType


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
class EvaluationFolderHierarchy:
    third_level: str
    fourth_level: Callable[[BaseSplitter], str]


@dataclass
class ScriptArgument:
    name: str
    value: Any
    default_value: Any
    dependency_name: str
    dependency_value: Any
    valid_arg_condition: bool
    error_condition_message: str


@dataclass
class DocSplitterOptions:
    doc_splitter_type: DocumentSplitterType
    add_title: bool
    token_count: int
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


@dataclass
class VQAStrategyDetail:
    country: str
    file_type: str
    vqa_strategy_type: VQAStrategyType
    prompt_type: PromptType
    doc_splitter_options: Optional[DocSplitterOptions] = None
