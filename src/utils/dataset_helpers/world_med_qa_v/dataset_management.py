import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

from src.utils.data_definitions import ModelAnswerResult, VQAStrategyDetail
from src.utils.enums import OutputFileType


def load_vqa_dataset(data_path: Path, country: str, file_type: str) -> Dataset:
    dataset_filename = f"{country}_{file_type}_processed.tsv"
    dataset_filepath = str(data_path / dataset_filename)

    print(f"- Loading WorldMedQA-V dataset (filename: {dataset_filename}) ...")
    dataset = load_dataset(
        "csv", 
        data_files=[dataset_filepath],
        sep="\t"
    )['train']
    print(f"+ WorldMedQA-V dataset (filename: {dataset_filename}) loaded.")
    return dataset


def get_dataset_row_by_id(
    dataset: Dataset,
    question_id: int
) -> dict:
    filtered_dataset = dataset.filter(lambda row: row['index'] == question_id)
    if len(filtered_dataset) == 0:
        raise ValueError(f"No row found with index {question_id}")
    return filtered_dataset[0]


def fetch_model_answer_from_json(
    evaluation_results_folder: Path,
    vqa_strategy_detail: VQAStrategyDetail,
    question_id: int,
) -> ModelAnswerResult:
    evaluation_results_filepath = vqa_strategy_detail.generate_output_filepath(
        root_folder=evaluation_results_folder, output_file_type=OutputFileType.JSON_FILE
    )

    with open(evaluation_results_filepath, mode='r', encoding='utf-8') as evaluation_file:
        evaluation_data = json.load(evaluation_file)

    return ModelAnswerResult(
        answer=evaluation_data['predictions'][str(question_id)]['predicted_answer']
    )


def load_evaluation_results(
    evaluation_results_folder: Path,
    vqa_strategy_details: list[VQAStrategyDetail]
) -> pd.DataFrame:
    evaluation_results = []

    for detail in vqa_strategy_details:
        evaluation_results_filepath = detail.generate_output_filepath(
            root_folder=evaluation_results_folder, output_file_type=OutputFileType.JSON_FILE
        )
        evaluation_metrics = __load_evaluation_result_from_filepath(evaluation_results_filepath)
        evaluation_results.append({
            "country": detail.country,
            "file_type": detail.file_type,
            "vqa_strategy_type": detail.vqa_strategy_type.value,
            "prompt_type": detail.prompt_type.value,
            "relevant_docs_count": detail.relevant_docs_count,
            "doc_splitter": (
                detail.doc_splitter_options.doc_splitter_type
                if detail.doc_splitter_options else None
            ),
            "add_title": (
                detail.doc_splitter_options.add_title if detail.doc_splitter_options else None
            ),
            "token_count": (
                detail.doc_splitter_options.token_count if detail.doc_splitter_options else None
            ),
            "chunk_size": (
                detail.doc_splitter_options.chunk_size if detail.doc_splitter_options else None
            ),
            "chunk_overlap": (
                detail.doc_splitter_options.chunk_overlap if detail.doc_splitter_options else None
            ),
            "should_apply_rag_to_question": detail.should_apply_rag_to_question,
            "accuracy": evaluation_metrics['accuracy'],
            "well_formatted_answers": evaluation_metrics['percentage_well_formatted']            
        })

    return pd.DataFrame(evaluation_results)


def get_max_accuracy_rows(evaluation_results: pd.DataFrame) -> pd.DataFrame:
    max_accuracy = evaluation_results['accuracy'].max()
    max_rows = evaluation_results[evaluation_results['accuracy'] == max_accuracy]
    return max_rows


def get_mean_accuracy(evaluation_results: pd.DataFrame) -> float:
    return evaluation_results['accuracy'].mean()


def get_subsets_data_by_split_type(
    splits_data: dict[str, Dataset],
    split_type: str
) -> dict[str, Any]:
    return dict(filter(lambda subset: subset[0].endswith(split_type), splits_data.items()))


def get_formatted_subset_prefix(subset_name: str) -> str:
    return subset_name.split('_')[0].capitalize()


# ====================
# Private Functions
# ====================


def __load_evaluation_result_from_filepath(
    evaluation_results_filepath: Path
) -> dict:
    with open(file=evaluation_results_filepath, mode='r', encoding='utf-8') as evaluation_file:
        evaluation_data = json.load(evaluation_file)

    return evaluation_data
