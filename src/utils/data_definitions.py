from dataclasses import dataclass, field, fields
from itertools import product
from pathlib import Path
from typing import Any, Optional, Type

from langchain_core.documents import Document

from src.utils.enums import (
    DocumentSplitterType,
    RagQPromptType,
    VQAStrategyType,
    ZeroShotPromptType
)
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
class ScriptArgument:
    name: str
    value: Any
    default_value: Any
    dependency_name: str
    dependency_value: Any
    valid_arg_condition: bool
    error_condition_message: str


@dataclass
class LoggerConfig:
    console_handler_enabled: bool
    file_handler_enabled: bool

    def __post_init__(self):
        self.__validate()

    def __validate(self) -> None:
        if not any(getattr(self, attribute.name) for attribute in fields(self)):
            raise ValueError("At least one option must be True")


class InvalidDocSplitterOptions(Exception):
    pass

@dataclass(frozen=True)
class DocSplitterOptions:
    doc_splitter_type: DocumentSplitterType
    token_count: int
    add_title: bool = False
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    def validate(self) -> None:
        requires_chunks = (
            self.doc_splitter_type == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER
        )

        if requires_chunks and (self.chunk_size is None or self.chunk_overlap is None):
            raise InvalidDocSplitterOptions(
                self.__invalid_chunk_message(
                    should_have_chunks=requires_chunks
                )
            )

        if not requires_chunks and (self.chunk_size is not None or self.chunk_overlap is not None):
            raise InvalidDocSplitterOptions(
                self.__invalid_chunk_message(
                    should_have_chunks=requires_chunks
                )
            )

    def __invalid_chunk_message(self, should_have_chunks: bool) -> str:
        action = "cannot be None" if should_have_chunks else "must be None"
        return (
            "If 'doc_splitter_type' is of type "
            f"{type(self.doc_splitter_type).__name__}.{self.doc_splitter_type.name}, "
            f"'chunk_size' and 'chunk_overlap' {action}"
        )


class InvalidVQAStrategyDetailError(Exception):
    pass

@dataclass(frozen=True)
class VQAStrategyDetail:
    country: str
    file_type: str
    vqa_strategy_type: VQAStrategyType
    prompt_type: PromptType
    relevant_docs_count: Optional[int] = None
    doc_splitter_options: Optional[DocSplitterOptions] = None
    should_apply_rag_to_question: Optional[bool] = None

    def __post_init__(self):
        self.__validate()

    def __validate(self) -> None:
        if self.vqa_strategy_type == VQAStrategyType.ZERO_SHOT:
            if not isinstance(self.prompt_type, ZeroShotPromptType):
                raise InvalidVQAStrategyDetailError(
                    self.__invalid_prompt_message(expected_type=ZeroShotPromptType)
                )

            if self.relevant_docs_count:
                raise InvalidVQAStrategyDetailError(
                    self.__invalid_field_message(field_name="relevant_docs_count")
                )

            if self.doc_splitter_options:
                raise InvalidVQAStrategyDetailError(
                    self.__invalid_field_message(field_name="doc_splitter_options")
                )

        if self.vqa_strategy_type in (VQAStrategyType.RAG_Q, VQAStrategyType.RAG_Q_AS):
            if not isinstance(self.prompt_type, RagQPromptType):
                raise InvalidVQAStrategyDetailError(
                    self.__invalid_prompt_message(expected_type=RagQPromptType)
                )

            if not self.relevant_docs_count:
                raise InvalidVQAStrategyDetailError(
                    self.__missing_field_message(field_name="relevant_docs_count")
                )

            if self.doc_splitter_options:
                try:
                    self.doc_splitter_options.validate()
                except InvalidDocSplitterOptions as e:
                    raise InvalidVQAStrategyDetailError(
                        f"'doc_splitter_type' has not been set up correctly. {e}"
                    ) from e

        if self.vqa_strategy_type in (VQAStrategyType.ZERO_SHOT, VQAStrategyType.RAG_Q):
            if self.should_apply_rag_to_question is not None:
                raise InvalidVQAStrategyDetailError(
                    self.__invalid_field_message(field_name="should_apply_rag_to_question")
                )

    def __invalid_prompt_message(self, expected_type: Type) -> str:
        return (
            "Invalid combination: "
            f"{type(self.vqa_strategy_type).__name__}.{self.vqa_strategy_type.name} with "
            f"{type(self.prompt_type).__name__}.{self.prompt_type.name}. "
            f"'prompt_type' should be of type {expected_type.__name__}"
        )

    def __invalid_field_message(self, field_name: str) -> str:
        return (
            "If 'vqa_strategy_type' is of type "
            f"{type(self.vqa_strategy_type).__name__}.{self.vqa_strategy_type.name}, "
            f"'{field_name}' must be None"
        )

    def __missing_field_message(self, field_name: str) -> str:
        return (
            "If 'vqa_strategy_type' is of type "
            f"{type(self.vqa_strategy_type).__name__}.{self.vqa_strategy_type.name}, "
            f"'{field_name}' cannot be None"
        )

    def generate_evaluation_results_filepath(self, evaluation_results_folder: Path) -> Path:

        def doc_splitter_attribute_to_path_elements(
            attribute_name: str,
            attribute_value: Any
        ) -> str:
            attribute_maps = {
                "doc_splitter_type": {
                    DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER: "rec_char_splitting",
                    DocumentSplitterType.PARAGRAPH_SPLITTER: "par_splitting",
                    DocumentSplitterType.SPACY_SENTENCE_SPLITTER: "spacy_sent_splitting"
                },
                "add_title": {
                    False: "no_title",
                    True: "with_title"
                }
            }

            if attribute_name in {"token_count", "chunk_size", "chunk_overlap"}:
                if attribute_value is not None:
                    short_attribute_name = "".join([word[0] for word in attribute_name.split('_')])
                    return f"{short_attribute_name}{attribute_value}"
                return ""

            if attribute_name in attribute_maps:
                return attribute_maps[attribute_name][attribute_value]

            raise ValueError(f"Attribute '{attribute_name}' not recognized")

        evaluation_results_filepath = evaluation_results_folder / self.vqa_strategy_type.value
        evaluation_results_filename = (
            f"{self.country}_{self.file_type}_{self.prompt_type.value}_evaluation.json"
        )

        relevant_docs_count_filepath = ""
        if self.relevant_docs_count:
            relevant_docs_count_filepath = f"rdc{self.relevant_docs_count}"

        rag_q_as_details_filepath = ""
        if self.vqa_strategy_type == VQAStrategyType.RAG_Q_AS:
            if self.should_apply_rag_to_question:
                rag_q_as_details_filepath = "question_and_answers"
            else:
                rag_q_as_details_filepath = "answers_only"

        doc_splitter_filepath = ""
        if self.vqa_strategy_type in (VQAStrategyType.RAG_Q, VQAStrategyType.RAG_Q_AS):
            doc_splitter_filepath = "no_doc_split"

        if self.doc_splitter_options:
            doc_splitter_attributes = {
                attribute.name: getattr(self.doc_splitter_options, attribute.name)
                for attribute in fields(self.doc_splitter_options)
            }
            doc_splitter_path_elements = [
                doc_splitter_attribute_to_path_elements(attribute_name, attribute_value)
                for attribute_name, attribute_value in doc_splitter_attributes.items()
            ]
            doc_splitter_filepath = Path(
                doc_splitter_path_elements[0],
                "_".join([element for element in doc_splitter_path_elements[1:] if element])
            )

        return Path(
            evaluation_results_filepath,
            relevant_docs_count_filepath,
            rag_q_as_details_filepath,
            doc_splitter_filepath,
            evaluation_results_filename
        )


@dataclass
class GeneralDocSplitterOptions:
    doc_splitter_types: list[DocumentSplitterType]
    token_counts: list[int]
    add_titles: list[bool]
    chunk_sizes: list[Optional[int]] = field(default_factory=list)
    chunk_overlaps: list[Optional[int]] = field(default_factory=list)


@dataclass
class GeneralVQAStrategiesDetails:
    countries: list[str]
    file_types: list[str]
    vqa_strategy_types: list[VQAStrategyType]
    prompt_types: list[PromptType]
    relevant_docs_count: list[Optional[int]] = field(default_factory=list)
    doc_splitter_options: list[Optional[GeneralDocSplitterOptions]] = field(default_factory=list)
    should_apply_rag_to_questions: list[Optional[bool]] = field(default_factory=list)

    def __post_init__(self):
        self.__validate()

    def __validate(self) -> None:
        if len(self.doc_splitter_options) > 2:
            raise ValueError("'doc_splitter_options' can contain at most two elements")

        none_count = 0
        doc_splitter_options_count = 0
        for option in self.doc_splitter_options:
            if not option:
                none_count += 1
            elif isinstance(option, GeneralDocSplitterOptions):
                doc_splitter_options_count += 1

        if none_count > 1 or doc_splitter_options_count > 1:
            raise ValueError((
                "'doc_splitter_options' can contain at most one 'None' and one "
                "'GeneralDocSplitterOptions'."
            ))

    def get_possible_vqa_strategy_details(self) -> list[VQAStrategyDetail]:
        doc_splitter_combinations = []
        for option in self.doc_splitter_options:
            if not option:
                doc_splitter_combinations.append(None)
            else:
                doc_splitter_combinations += list(product(
                    *[getattr(option, attribute.name) for attribute in fields(option)]
                ))

        all_attributes = [
            getattr(self, attribute.name)
            for attribute in fields(self) if attribute.name != 'doc_splitter_options'
        ]
        all_combinations = list(product(*all_attributes, doc_splitter_combinations))

        possible_combinations = []
        for combination in all_combinations:
            try:
                vqa_strategy_detail = VQAStrategyDetail(
                    country=combination[0],
                    file_type=combination[1],
                    vqa_strategy_type=combination[2],
                    prompt_type=combination[3],
                    relevant_docs_count=combination[4],
                    doc_splitter_options=DocSplitterOptions(
                        doc_splitter_type=combination[6][0],
                        token_count=combination[6][1],
                        add_title=combination[6][2],
                        chunk_size=combination[6][3],
                        chunk_overlap=combination[6][4]
                    ) if combination[6] else None,
                    should_apply_rag_to_question=combination[5]
                )
                possible_combinations.append(vqa_strategy_detail)
            except InvalidVQAStrategyDetailError as e:
                print(f"Invalid VQAStrategyDetail: {e}. Skipping...")
                continue

        unique_combinations = list(dict.fromkeys(possible_combinations))
        return unique_combinations
