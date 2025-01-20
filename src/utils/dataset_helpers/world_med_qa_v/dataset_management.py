import json
from pathlib import Path

from datasets import Dataset, load_dataset

from src.utils.data_definitions import ModelAnswerResult


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
    vqa_strategy_name: str,
    country: str,
    file_type: str,
    prompt_type_name: str,
    question_id: int,
) -> ModelAnswerResult:
    evaluation_results_filename = f'{country}_{file_type}_{prompt_type_name}_evaluation.json'
    evaluation_results_path_elements = [
        evaluation_results_folder,
        vqa_strategy_name,
        evaluation_results_filename
    ]
    evaluation_results_path = Path(*evaluation_results_path_elements)
    with open(evaluation_results_path, mode='r', encoding='utf-8') as evaluation_file:
        evaluation_data = json.load(evaluation_file)

    return ModelAnswerResult(
        answer=evaluation_data['predictions'][str(question_id)]['predicted_answer']
    )
