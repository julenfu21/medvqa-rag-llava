from pathlib import Path
from datasets import Dataset, load_dataset


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
