import argparse
from pathlib import Path

from src.utils.data_definitions import ScriptArgument
from src.utils.dataset_helpers.world_med_qa_v.dataset_management import load_vqa_dataset
from src.utils.enums import (
    DocumentSplitterType,
    RagQPromptType,
    VQAStrategyType,
    ZeroShotPromptType
)
from src.utils.text_splitters.base_splitter import BaseSplitter
from src.utils.text_splitters.paragraph_splitter import ParagraphSplitter
from src.utils.text_splitters.recursive_character_splitter import RecursiveCharacterSplitter
from src.utils.text_splitters.spacy_sentence_splitter import SpacySentenceSplitter
from src.utils.types_aliases import PromptType
from src.visual_qa_model import VisualQAModel
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy
from src.visual_qa_strategies.rag_q_as_vqa_strategy import RagQAsVQAStrategy
from src.visual_qa_strategies.rag_q_vqa_strategy import RagQVQAStrategy
from src.visual_qa_strategies.zero_shot_vqa_strategy import ZeroShotVQAStrategy


OLLAMA_MODEL_NAME = "llava"

DEFAULT_INDEX_DIR = Path("data/WikiMed/indexed_db")
DEFAULT_INDEX_NAME = "Wikimed+S-PubMedBert-MS-MARCO-FullTexts"
DEFAULT_EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
DEFAULT_RELEVANT_DOCS_COUNT = 1
DEFAULT_TOKEN_COUNT = {
    DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER: 2,
    DocumentSplitterType.SPACY_SENTENCE_SPLITTER: 2,
    DocumentSplitterType.PARAGRAPH_SPLITTER: 1
}
DEFAULT_ADD_TITLE = False
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0



def vqa_strategy_type_to_prompt_type(vqa_strategy: VQAStrategyType, prompt_name: str) -> PromptType:
    if vqa_strategy == VQAStrategyType.ZERO_SHOT:
        return ZeroShotPromptType(prompt_name)

    if vqa_strategy in{VQAStrategyType.RAG_Q, VQAStrategyType.RAG_Q_AS}:
        return RagQPromptType(prompt_name)

    raise TypeError("Unhandled VQA strategy type")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Script to run evaluation of the LLaVA model over the WorldMedQA-V leveraging the RAG "
            "strategy chosen."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Display argument values before execution")

    parser.add_argument("--dataset_dir",
                        type=Path, default=Path("data/WorldMedQA-V"),
                        help="Directory containing the dataset")
    parser.add_argument("--country",
                        type=str, default="spain",
                        choices=["brazil", "israel", "japan", "spain"],
                        help="Origin country of the dataset")
    parser.add_argument("--file_type",
                        type=str, default="english",
                        choices=["english", "local"],
                        help="Language of the questions (original or English translation)")
    parser.add_argument("--results_dir",
                        type=Path, default=Path("evaluation_results"),
                        help="Directory that contains the evaluation results")
    parser.add_argument("--vqa_strategy",
                        type=VQAStrategyType, required=True,
                        choices=list(VQAStrategyType),
                        help="VQA strategy used to modify the input of the model")
    parser.add_argument("--prompt_type",
                        type=str, required=True,
                        help=(
                            "Prompt type, determined by --vqa-strategy\n"
                            "    - Options for 'zero_shot': "
                            f"{[prompt_name.value for prompt_name in ZeroShotPromptType]}\n"
                            "    - Options for 'rag_q': "
                            f"{[prompt_name.value for prompt_name in RagQPromptType]}"
                        ))

    # Arguments used when '--vqa_strategy' != 'zero_shot'
    parser.add_argument("--index_dir",
                        type=Path, default=None,
                        help=(
                            "Directory that stores the indexed dataset leveraged to apply RAG\n"
                            "    * It can only be used if --vqa_strategy is not 'zero_shot'\n"
                            "    * If not provided, the default value will be "
                            f"'{DEFAULT_INDEX_DIR}'"
                        ))
    parser.add_argument("--index_name",
                        type=str, default=None,
                        help=(
                            "Name of the index\n"
                            "    * It can only be used if --vqa_strategy is not 'zero_shot'\n"
                            "    * If not provided, the default value will be "
                            f"'{DEFAULT_INDEX_NAME}'"
                        ))
    parser.add_argument("--embedding_model_name",
                        type=str, default=None,
                        help=(
                            "Name of the embedding model\n"
                            "    * It can only be used if --vqa_strategy is not 'zero_shot'\n"
                            "    * If not provided, the default value will be "
                            f"'{DEFAULT_EMBEDDING_MODEL_NAME}'"
                        ))
    parser.add_argument("--relevant_docs_count",
                        type=int, default=None,
                        help=(
                            "Amount or documents to be added to the input of the model when "
                            "applying RAG\n"
                            "    * It can only be used if --vqa_strategy is not 'zero_shot'\n"
                            "    * If not provided, the default value will be "
                            f"'{DEFAULT_RELEVANT_DOCS_COUNT}'"
                        ))
    parser.add_argument("--should_apply_rag_to_question",
                        action='store_true', default=None,
                        help=(
                            "Apply RAG also to the question apart from the answer.\n"
                            "    * It can only be used if --vqa_strategy is 'rag_q_as'\n"
                            "    * If not provided, the default value will be 'None'"
                        ))
    parser.add_argument("--doc_splitter",
                        type=DocumentSplitterType, default=None,
                        choices=list(DocumentSplitterType),
                        help=(
                            "Document splitter used to split the documents used for RAG into "
                            "smaller chunks\n"
                            "    * It can only be used if --vqa_strategy is not 'zero_shot'\n"
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

    args.prompt_type = vqa_strategy_type_to_prompt_type(
        vqa_strategy=args.vqa_strategy,
        prompt_name=args.prompt_type
    )

    conditional_arguments = [
        ScriptArgument(
            name="index_dir",
            value=args.index_dir,
            default_value=DEFAULT_INDEX_DIR,
            dependency_name="vqa_strategy",
            dependency_value=args.vqa_strategy,
            valid_arg_condition=args.vqa_strategy != VQAStrategyType.ZERO_SHOT,
            error_condition_message="--vqa_strategy is 'zero_shot'"
        ),
        ScriptArgument(
            name="index_name",
            value=args.index_name,
            default_value=DEFAULT_INDEX_NAME,
            dependency_name="vqa_strategy",
            dependency_value=args.vqa_strategy,
            valid_arg_condition=args.vqa_strategy != VQAStrategyType.ZERO_SHOT,
            error_condition_message="--vqa_strategy is 'zero_shot'"
        ),
        ScriptArgument(
            name="embedding_model_name",
            value=args.embedding_model_name,
            default_value=DEFAULT_EMBEDDING_MODEL_NAME,
            dependency_name="vqa_strategy",
            dependency_value=args.vqa_strategy,
            valid_arg_condition=args.vqa_strategy != VQAStrategyType.ZERO_SHOT,
            error_condition_message="--vqa_strategy is 'zero_shot'"
        ),
        ScriptArgument(
            name="relevant_docs_count",
            value=args.relevant_docs_count,
            default_value=DEFAULT_RELEVANT_DOCS_COUNT,
            dependency_name="vqa_strategy",
            dependency_value=args.vqa_strategy,
            valid_arg_condition=args.vqa_strategy != VQAStrategyType.ZERO_SHOT,
            error_condition_message="--vqa_strategy is 'zero_shot'"
        ),
        ScriptArgument(
            name="doc_splitter",
            value=args.doc_splitter,
            default_value=None,
            dependency_name="vqa_strategy",
            dependency_value=args.vqa_strategy,
            valid_arg_condition=args.vqa_strategy != VQAStrategyType.ZERO_SHOT,
            error_condition_message="--vqa_strategy is 'zero_shot'"
        ),
        ScriptArgument(
            name="should_apply_rag_to_question",
            value=args.should_apply_rag_to_question,
            default_value=None,
            dependency_name="vqa_strategy",
            dependency_value=args.vqa_strategy,
            valid_arg_condition=args.vqa_strategy == VQAStrategyType.RAG_Q_AS,
            error_condition_message="--vqa_strategy is not 'rag_q_as'"
        ),
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


def get_strategy(
    arguments: argparse.Namespace
) -> BaseVQAStrategy:
    match arguments.vqa_strategy:
        case VQAStrategyType.ZERO_SHOT:
            return ZeroShotVQAStrategy(prompt_type=arguments.prompt_type)
        case VQAStrategyType.RAG_Q:
            return RagQVQAStrategy(
                prompt_type=arguments.prompt_type,
                index_dir=arguments.index_dir,
                index_name=arguments.index_name,
                embedding_model_name=arguments.embedding_model_name,
                relevant_docs_count=arguments.relevant_docs_count
            )
        case VQAStrategyType.RAG_Q_AS:
            return RagQAsVQAStrategy(
                prompt_type=arguments.prompt_type,
                index_dir=arguments.index_dir,
                index_name=arguments.index_name,
                embedding_model_name=arguments.embedding_model_name,
                relevant_docs_count=arguments.relevant_docs_count
            )
        # case VQAStrategyType.RAG_Q_AS:
        #     return RagQAsVQAStrategy(prompt_type=None)
        # case VQAStrategyType.RAG_IMG:
        #     return RagImgVQAStrategy(prompt_type=None)
        # case VQAStrategyType.RAG_DB_RERANKER:
        #     return RagDBRerankerVQAStrategy(prompt_type=None)

    raise TypeError("Unhandled VQA strategy type")


def get_document_splitter(
    arguments: argparse.Namespace
) -> BaseSplitter:
    match arguments.doc_splitter:
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


def main() -> None:
    args = parse_args()

    if args.verbose:
        print("\nArguments received:")
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                print(f"\t--{arg_name} ({type(arg_value).__name__}): {arg_value}")
        print()

    world_med_qa_v_dataset = load_vqa_dataset(
        data_path=args.dataset_dir,
        country=args.country,
        file_type=args.file_type
    )
    llava_model = VisualQAModel(
        visual_qa_strategy=get_strategy(arguments=args),
        model_name=OLLAMA_MODEL_NAME,
        country=args.country,
        file_type=args.file_type
    )
    llava_model.evaluate(
        dataset=world_med_qa_v_dataset,
        results_path=args.results_dir,
        doc_splitter=get_document_splitter(arguments=args),
        should_apply_rag_to_question=args.should_apply_rag_to_question
    )


if __name__ == "__main__":
    main()
