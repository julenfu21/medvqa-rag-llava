import argparse
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document

import src.utils.dataset_helpers.wikimed.dataset_management as wikimed_dataset_management
from scripts.utils.script_helpers import get_document_splitter, process_conditional_argument
from scripts.utils.contants import (
    DEFAULT_ADD_TITLE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOKEN_COUNT
)
from src.utils.data_definitions import ScriptArgument
from src.utils.enums import DocumentSplitterType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Script to visualize a document from the WikiMed Dataset and apply various Document "
            "Splitting Techniques to see how the content would be segmented before being passed "
            "to a model."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Display argument values before execution")

    parser.add_argument("--dataset_dir",
                        type=Path, default=Path("data/WikiMed/WikiMed.json"),
                        help="Directory containing the dataset")
    parser.add_argument("--document_title",
                        type=str, required=True,
                        help="Title of the document in the dataset")
    parser.add_argument("--full_text_char_limit",
                        type=int, default=None,
                        help=(
                            "Maximum length of the full text to be displayed as the result. "
                            "If the full text exceeds this value, it will be truncated to "
                            "fit and '...' will be appended (the ellipsis is not included in "
                            "the limit)."
                        ))
    parser.add_argument("--doc_splitter",
                        type=DocumentSplitterType, default=None,
                        choices=list(DocumentSplitterType),
                        help=(
                            "Document splitter used to split the documents used for RAG into "
                            "smaller chunks\n"
                            "    * If not provided, the default value will be 'None'"
                        ))

    # Arguments used when '--doc_splitter' != None
    parser.add_argument("--token_count",
                        type=int, default=None,
                        help=(
                            "Number of chunks to extract from split documents\n"
                            "    * It can only be used if --doc_splitter is not 'None'\n"
                            "    * If not provided, the default value will be determined by "
                            "--doc_splitter:\n"
                            "        - 'recursive_character_splitter': "
                            f"{DEFAULT_TOKEN_COUNT[DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER]}\n"
                            "        - 'spacy_sentence_splitter': "
                            f"{DEFAULT_TOKEN_COUNT[DocumentSplitterType.SPACY_SENTENCE_SPLITTER]}\n"
                            "        - 'paragraph_splitter': "
                            f"{DEFAULT_TOKEN_COUNT[DocumentSplitterType.PARAGRAPH_SPLITTER]}"
                        ))
    parser.add_argument("--add_title",
                        action='store_true', default=None,
                        help=(
                            "Include the document title in RAG inputs.\n"
                            "    * It can only be used if --doc_splitter is not 'None'\n"
                            "    * If not provided, the default value will be "
                            f"'{DEFAULT_ADD_TITLE}'"
                        ))

    # Arguments used when '--doc_splitter' == 'recursive_character_splitter'
    parser.add_argument("--chunk_size",
                        type=int, default=None,
                        help=(
                            "Maximum size of chunks to extract from split documents\n"
                            "    * It can only be used if --doc_splitter is "
                            "'recursive_character_splitter'\n"
                            "    * If not provided, the default value will be "
                            f"'{DEFAULT_CHUNK_SIZE}'"
                        ))
    parser.add_argument("--chunk_overlap",
                        type=int, default=None,
                        help=(
                            "Overlap in characters between chunks\n"
                            "    * It can only be used if --doc_splitter is "
                            "'recursive_character_splitter'\n"
                            "    * If not provided, the default value will be "
                            f"{DEFAULT_CHUNK_OVERLAP}"
                        ))

    args = parser.parse_args()

    conditional_arguments = [
        ScriptArgument(
            name="token_count",
            value=args.token_count,
            default_value=DEFAULT_TOKEN_COUNT,
            dependency_name="doc_splitter",
            dependency_value=args.doc_splitter,
            valid_arg_condition=args.doc_splitter is not None,
            error_condition_message="--doc_splitter is None"
        ),
        ScriptArgument(
            name="add_title",
            value=args.add_title,
            default_value=DEFAULT_ADD_TITLE,
            dependency_name="doc_splitter",
            dependency_value=args.doc_splitter,
            valid_arg_condition=args.doc_splitter is not None,
            error_condition_message="--doc_splitter is None"
        ),
        ScriptArgument(
            name="chunk_size",
            value=args.chunk_size,
            default_value=DEFAULT_CHUNK_SIZE,
            dependency_name="doc_splitter",
            dependency_value=args.doc_splitter,
            valid_arg_condition=(
                args.doc_splitter == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER
            ),
            error_condition_message="--doc_splitter is not 'recursive_character_splitter'"
        ),
        ScriptArgument(
            name="chunk_overlap",
            value=args.chunk_overlap,
            default_value=DEFAULT_CHUNK_OVERLAP,
            dependency_name="doc_splitter",
            dependency_value=args.doc_splitter,
            valid_arg_condition=(
                args.doc_splitter == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER
            ),
            error_condition_message="--doc_splitter is not 'recursive_character_splitter'"
        )
    ]

    for argument in conditional_arguments:
        process_conditional_argument(
            parser=parser,
            args=args,
            argument=argument
        )

    return args


def pretty_print_result(
    document_id: str,
    title: str,
    text: str,
    full_text_char_limit: Optional[int],
    split_text: Optional[str] = None
) -> None:
    print()
    print(f"Document ID: {document_id}")
    print(f"Document Title: {title}")
    full_text_section_title = (
        f'Full Text (first {full_text_char_limit} chars)'
        if full_text_char_limit else 'Full Text'
    )
    pretty_print_section(
        title=full_text_section_title,
        variable=text,
        char_limit=full_text_char_limit
    )
    if split_text:
        pretty_print_section(title='Split Text', variable=split_text)

def pretty_print_section(
    title: str,
    variable: Any,
    padding: int = 4,
    char_limit: Optional[int] = None
) -> None:
    box_width = len(title) + padding * 2 + 2
    top_bottom_row = "#" * box_width
    middle_row = "#" + " " * (box_width - 2) + "#"

    title_str = title.center(box_width - 2)
    title_row = f"#{title_str}#"

    print()
    print(top_bottom_row)
    print(middle_row)
    print(title_row)
    print(middle_row)
    print(top_bottom_row)
    print()
    if char_limit and len(variable) > char_limit:
        print(f"{variable[:char_limit]} ...")
    else:
        print(variable)
    print()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print("\nArguments received:")
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                print(f"\t--{arg_name} ({type(arg_value).__name__}): {arg_value}")
        print()

    dataset_metadata = wikimed_dataset_management.load_wikimed_dataset_metadata(
        data_path=args.dataset_dir
    )

    try:
        row = wikimed_dataset_management.get_dataset_row_by_doc_title(
            dataset_path=args.dataset_dir,
            dataset_metadata=dataset_metadata,
            doc_title=args.document_title
        )
    except ValueError:
        print((
            f"\nUnable to find a document with title: {args.document_title}"
            "\n"
            "Make sure a document with the title entered exists."
        ))
    else:
        document_splitter = get_document_splitter(arguments=args)
        document = Document(page_content=row['text'])

        if document_splitter:
            split_document_text = document_splitter.split_documents(documents=[document])[0]
        else:
            split_document_text = None

        pretty_print_result(
            document_id=row['_id'],
            title=row['title'],
            text=row['text'],
            full_text_char_limit=args.full_text_char_limit,
            split_text=split_document_text
        )


if __name__ == '__main__':
    main()
