import argparse

from src.utils.data_definitions import ScriptArgument
from src.utils.enums import DocumentSplitterType
from src.utils.text_splitters.base_splitter import BaseSplitter
from src.utils.text_splitters.no_splitter import NoSplitter
from src.utils.text_splitters.paragraph_splitter import ParagraphSplitter
from src.utils.text_splitters.recursive_character_splitter import RecursiveCharacterSplitter
from src.utils.text_splitters.spacy_sentence_splitter import SpacySentenceSplitter


def process_conditional_argument(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argument: ScriptArgument
) -> None:
    if not argument.valid_arg_condition and argument.value is not None:
        parser.error(
            f"--{argument.name} should not be provided when {argument.error_condition_message}"
        )

    if argument.valid_arg_condition and argument.value is None:
        if isinstance(argument.default_value, dict):
            default_value = argument.default_value[argument.dependency_value]
            default_value_message = default_value_message = (
                f"--{argument.name} was not provided. Using default value "
                f"(--{argument.dependency_name}='{argument.dependency_value}'): '{default_value}'"
            )
        else:
            default_value = argument.default_value
            default_value_message = (
                f"--{argument.name} was not provided. Using default value: "
                f"'{default_value}'"
            )

        print(default_value_message)
        setattr(args, argument.name, default_value)


def get_document_splitter(
    arguments: argparse.Namespace
) -> BaseSplitter:
    match arguments.doc_splitter:
        case DocumentSplitterType.NO_SPLITTER:
            return NoSplitter(
                token_count=-1,
                add_title=arguments.add_title
            )
        case DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER:
            return RecursiveCharacterSplitter(
                token_count=arguments.token_count,
                chunk_size=arguments.chunk_size,
                chunk_overlap=arguments.chunk_overlap,
                add_title=arguments.add_title
            )
        case DocumentSplitterType.SPACY_SENTENCE_SPLITTER:
            return SpacySentenceSplitter(
                token_count=arguments.token_count,
                add_title=arguments.add_title
            )
        case DocumentSplitterType.PARAGRAPH_SPLITTER:
            return ParagraphSplitter(
                token_count=arguments.token_count,
                add_title=arguments.add_title
            )

    return None
