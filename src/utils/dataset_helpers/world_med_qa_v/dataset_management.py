import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

from src.utils.data_definitions import ModelAnswerResult, VQAStrategyDetail
from src.utils.enums import DocumentSplitterType, VQAStrategyType


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


def load_evaluation_results(
    evaluation_results_folder: Path,
    vqa_strategy_details: list[VQAStrategyDetail]
) -> pd.DataFrame:
    evaluation_results = []
    doc_splitter_type_to_folder_name = {
        DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER: 'rec_char_splitting',
        DocumentSplitterType.PARAGRAPH_SPLITTER: 'par_splitting',
        DocumentSplitterType.SPACY_SENTENCE_SPLITTER: 'spacy_sent_splitting',
        None: 'no_doc_split'
    }

    for detail in vqa_strategy_details:
        evaluation_results_filename = (
            f'{detail.country}_{detail.file_type}_{detail.prompt_type.value}_evaluation.json'
        )
        extra_path_elements = []
        document_splitting_details = []
        evaluation_results_path_elements = []

        if detail.vqa_strategy_type == VQAStrategyType.RAG_Q:
            if detail.doc_splitter_options:
                doc_splitter_folder_name = doc_splitter_type_to_folder_name[
                    detail.doc_splitter_options.doc_splitter_type
                ]

                if detail.doc_splitter_options.doc_splitter_type is not None:
                    add_title = (
                        'with_title' if detail.doc_splitter_options.add_title else 'no_title'
                    )
                    token_count = f"tc{detail.doc_splitter_options.token_count}"
                    document_splitting_details = [add_title, token_count]

                    if detail.doc_splitter_options.doc_splitter_type == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER:
                        chunk_size = f"cs{detail.doc_splitter_options.chunk_size}"
                        chunk_overlap = f"co{detail.doc_splitter_options.chunk_overlap}"
                        document_splitting_details.extend([chunk_size, chunk_overlap])
                    elif detail.doc_splitter_options.doc_splitter_type == DocumentSplitterType.SPACY_SENTENCE_SPLITTER:
                        model_name = 'en_core_web_sm'
                        document_splitting_details.extend([model_name])
            else:
                doc_splitter_folder_name = doc_splitter_type_to_folder_name[None]

            extra_path_elements = [
                doc_splitter_folder_name,
                "_".join(document_splitting_details)
            ]
            evaluation_results_path_elements = [
                evaluation_results_folder,
                detail.vqa_strategy_type.value,
                f"rdc{detail.relevant_docs_count}",
                *extra_path_elements,
                evaluation_results_filename
            ]
        else:
            evaluation_results_path_elements = [
                evaluation_results_folder,
                detail.vqa_strategy_type.value,
                evaluation_results_filename
            ]
        evaluation_results_filepath = Path(*evaluation_results_path_elements)

        evaluation_metrics = __load_evaluation_result_from_filepath(evaluation_results_filepath)
        evaluation_results.append({
            "country": detail.country,
            "file_type": detail.file_type,
            "vqa_strategy_type": detail.vqa_strategy_type.value,
            "prompt_type": detail.prompt_type.value,
            "doc_splitter": doc_splitter_type_to_folder_name[
                detail.doc_splitter_options.doc_splitter_type
                if detail.doc_splitter_options else None
            ],
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
            "accuracy": evaluation_metrics['accuracy'],
            "well_formatted_answers": evaluation_metrics['percentage_well_formatted']            
        })

    return pd.DataFrame(evaluation_results)


# ====================
# Private Functions
# ====================


def __load_evaluation_result_from_filepath(
    evaluation_results_filepath: Path
) -> dict:
    with open(file=evaluation_results_filepath, mode='r', encoding='utf-8') as evaluation_file:
        evaluation_data = json.load(evaluation_file)

    return evaluation_data
