import argparse
import itertools
from pathlib import Path
from typing import get_args

import src.utils.dataset_helpers.world_med_qa_v.dataset_management as world_med_qa_v_dataset_management
from src.utils.enums import RagQPromptType, VQAStrategyType, ZeroShotPromptType
from src.utils.types import PromptType
from src.visual_qa_model import VisualQAModel
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy
from src.visual_qa_strategies.rag_db_reranker_vqa_strategy import RagDBRerankerVQAStrategy
from src.visual_qa_strategies.rag_img_vqa_strategy import RagImgVQAStrategy
from src.visual_qa_strategies.rag_q_as_vqa_strategy import RagQAsVQAStrategy
from src.visual_qa_strategies.rag_q_vqa_strategy import RagQVQAStrategy
from src.visual_qa_strategies.zero_shot_vqa_strategy import ZeroShotVQAStrategy


OLLAMA_MODEL_NAME = "llava"
PROMPT_TYPES = get_args(PromptType)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Script to run evaluation of the Llava model over the WorldMedQA-V "
            "leveraging the RAG strategy chosen."
        ))
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
                        type=VQAStrategyType, default=VQAStrategyType.ZERO_SHOT,
                        choices=list(VQAStrategyType),
                        help="VQA strategy used to modify the input of the model")
    parser.add_argument("--prompt_type",
                        type=parse_prompt_type, default=ZeroShotPromptType.V1,
                        choices=[item for prompt_type in PROMPT_TYPES for item in list(prompt_type)],
                        help="Prompt used alongside the model to try to guide its behaviour")

    parser.add_argument("--index_dir",
                        type=Path, default=Path("data/WikiMed/indexed_db"),
                        help="Directory that stores the indexed dataset leveraged to apply RAG")
    parser.add_argument("--index_name",
                        type=str, default="Wikimed+S-PubMedBert-MS-MARCO-FullTexts",
                        help="Name of the index")
    parser.add_argument("--embedding_model_name",
                        type=str, default="pritamdeka/S-PubMedBert-MS-MARCO",
                        help="Name of the embedding model")
    parser.add_argument("--relevant_docs_count",
                        type=int, default=1,
                        help=(
                            "Amount or documents to be added to the input of the model when "
                            "applying RAG"
                        ))
    return parser.parse_args()


def parse_prompt_type(value: str) -> PromptType:
    for enum_class in PROMPT_TYPES:
        try:
            return enum_class(value)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid prompt type: {value}. Valid options are: "
        f"{[item for prompt_type in PROMPT_TYPES for item in list(prompt_type)]}"
    )


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
            return RagQAsVQAStrategy(prompt_type=None)
        case VQAStrategyType.RAG_IMG:
            return RagImgVQAStrategy(prompt_type=None)
        case VQAStrategyType.RAG_DB_RERANKER:
            return RagDBRerankerVQAStrategy(prompt_type=None)

    raise ValueError(
        f"Unhandled VQA strategy type: {arguments.vqa_strategy}. Valid options are: "
        f"{list(itertools.chain(ZeroShotPromptType, RagQPromptType))}"
    )


def main() -> None:
    args = parse_args()

    world_med_qa_v_dataset = world_med_qa_v_dataset_management.load_vqa_dataset(
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
        dataset=world_med_qa_v_dataset.take(1),
        save_path=args.results_dir
    )


if __name__ == "__main__":
    main()
