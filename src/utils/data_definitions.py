from dataclasses import dataclass, field, fields
from itertools import product
from pathlib import Path
from typing import Any, Callable, Optional, Type

from langchain_core.documents import Document

from src.utils.enums import (
    DocumentSplitterType,
    RagQPromptType,
    VQAStrategyType,
    ZeroShotPromptType
)
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


class InvalidDocSplitterOptions(Exception):
    pass

@dataclass(frozen=True)
class DocSplitterOptions:
    doc_splitter_type: Optional[DocumentSplitterType] = None
    add_title: Optional[bool] = None
    token_count: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

    def validate(self) -> None:
        if not self.doc_splitter_type:
            rest_of_attributes = [
                getattr(self, attribute.name) for attribute in fields(self)
                if attribute.name != 'doc_splitter_type'
            ]
            if any(attr is not None for attr in rest_of_attributes):
                raise InvalidDocSplitterOptions(
                    "If 'doc_splitter_type' is None, the rest of the fields must also be None"
                )
        else:
            if self.add_title is None:
                raise InvalidDocSplitterOptions(
                    self.__missing_field_message(field_name="add_title")
                )

            if not self.token_count:
                raise InvalidDocSplitterOptions(
                    self.__missing_field_message(field_name="token_count")
                )

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

    def __missing_field_message(self, field_name: str) -> str:
        return f"If 'doc_splitter_type' is not None, {field_name} cannot be None."


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

    def __post_init__(self):
        self.__validate()

    def __validate(self) -> None:
        if self.vqa_strategy_type == VQAStrategyType.ZERO_SHOT:
            if not isinstance(self.prompt_type, ZeroShotPromptType):
                raise InvalidVQAStrategyDetailError(
                    self.__invalid_prompt_message(expected_type=ZeroShotPromptType)
                )

            if self.relevant_docs_count:
                raise InvalidVQAStrategyDetailError((
                    self.__invalid_field_message(field_name="relevant_docs_count")
                ))

            if self.doc_splitter_options:
                raise InvalidVQAStrategyDetailError((
                    self.__invalid_field_message(field_name="doc_splitter_options")
                ))

        elif self.vqa_strategy_type == VQAStrategyType.RAG_Q:
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
        evaluation_results_filepath = evaluation_results_folder / self.vqa_strategy_type.value
        evaluation_results_filename = (
            f"{self.country}_{self.file_type}_{self.prompt_type.value}_evaluation.json"
        )

        def doc_splitter_attribute_to_path_elements(
            attribute_name: str,
            attribute_value: Any
        ) -> str:
            attribute_maps = {
                "doc_splitter_type": {
                    DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER: "rec_char_splitting",
                    DocumentSplitterType.PARAGRAPH_SPLITTER: "par_splitting",
                    DocumentSplitterType.SPACY_SENTENCE_SPLITTER: "spacy_sent_splitting",
                    None: "no_doc_split",
                },
                "add_title": {
                    False: "no_title",
                    True: "with_title",
                    None: "",
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

        doc_splitter_filepath = ""
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
                f"rdc{self.relevant_docs_count}",
                doc_splitter_path_elements[0],
                "_".join([element for element in doc_splitter_path_elements[1:] if element])
            )

        return Path(
            evaluation_results_filepath,
            doc_splitter_filepath,
            evaluation_results_filename
        )


@dataclass
class GeneralDocSplitterOptions:
    doc_splitter_types: list[Optional[DocumentSplitterType]] = field(default_factory=list)
    add_titles: list[bool] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    chunk_sizes: list[Optional[int]] = field(default_factory=list)
    chunk_overlaps: list[Optional[int]] = field(default_factory=list)


@dataclass
class GeneralVQAStrategiesDetails:
    countries: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    vqa_strategy_types: list[VQAStrategyType] = field(default_factory=list)
    prompt_types: list[PromptType] = field(default_factory=list)
    relevant_docs_count: list[Optional[int]] = field(default_factory=list)
    doc_splitter_options: GeneralDocSplitterOptions = (
        field(default_factory=GeneralDocSplitterOptions)
    )

    def __post_init__(self):
        # self.__add_default_values()
        pass

    # def __add_default_values(self) -> None:
    #     if not self.vqa_strategy_types:
    #         # COGER TODOS DEL ENUM (si archivo no existe imprimir aviso de que el archivo no existe)
    #         self.vqa_strategy_types = [VQAStrategyType.ZERO_SHOT, VQAStrategyType.RAG_Q]

    #     if not self.prompt_types:
    #         self.prompt_types = [
    #             prompt_value
    #             for prompt_type in get_args(PromptType)
    #             for prompt_value in list(prompt_type)
    #         ]

    #     if not self.relevant_docs_count:
    #         self.relevant_docs_count = [1, 2, 3]
    #     if (
    #         VQAStrategyType.ZERO_SHOT in self.vqa_strategy_types and
    #         None not in self.relevant_docs_count
    #     ):
    #         self.relevant_docs_count += [None]

        # if not self.doc_splitter_options:
        #     self.doc_splitter_options = [
        #         None,
        #         # DocSplitterOptions(
        #               AÑADIR TODOS POR DEFECTO
        #         # )
        #     ]
        # if (
        #     VQAStrategyType.RAG_Q in self.vqa_strategy_types and
        #     None not in self.doc_splitter_options.doc_splitter_types
        # ):
        #     self.doc_splitter_options.doc_splitter_types += [None]

    def get_possible_vqa_strategy_details(self) -> list[VQAStrategyDetail]:
        flat_doc_splitter_attributes = [
            getattr(self.doc_splitter_options, attribute.name)
            for attribute in fields(self.doc_splitter_options)
        ]
        rest_of_the_attributes = [getattr(self, attribute.name) for attribute in fields(self)[:-1]]
        all_attributes = rest_of_the_attributes + flat_doc_splitter_attributes
        for attribute in all_attributes:
            print(f"{attribute=}")
        all_combinations = list(product(*all_attributes))
        # for combination_id, combination in enumerate(all_combinations, start=1):
        #     print(f"{combination_id} --> {combination}")

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
                        doc_splitter_type=combination[5],
                        add_title=combination[6],
                        token_count=combination[7],
                        chunk_size=combination[8],
                        chunk_overlap=combination[9]
                    ) if combination[2] != VQAStrategyType.ZERO_SHOT else None
                )
                possible_combinations.append(vqa_strategy_detail)
            except InvalidVQAStrategyDetailError as e:
                print(f"Invalid VQAStrategyDetail: {e}. Skipping...") # ADD LOG
                continue

        unique_combinations = list(dict.fromkeys(possible_combinations))
        for combination_id, combination in enumerate(unique_combinations, start=1):
            print(f"{combination_id} --> {combination}")
        return unique_combinations
