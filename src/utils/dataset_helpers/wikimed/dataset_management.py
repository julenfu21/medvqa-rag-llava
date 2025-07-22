import json
import linecache
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_ollama import ChatOllama
from tqdm import tqdm


def load_wikimed_dataset_metadata(data_path: Path) -> pd.DataFrame:
    wikimed_dataset_metadata = []
    llava_model = ChatOllama(model="llava", temperature=0, num_predict=1)

    with open(file=data_path, mode="r", encoding="utf-8") as wikimed_file:
        wikimed_file.seek(0, 2)
        wikimed_length = wikimed_file.tell()
        wikimed_file.seek(0)

        with tqdm(
            total=wikimed_length,
            desc="- Loading WikiMed dataset metadata ...",
            unit="B",
            unit_scale=True
        ) as progress_bar:
            for line in wikimed_file:
                line_content = json.loads(line)
                line_id = line_content['_id']
                line_title = line_content['title']
                line_text = line_content['text']
                model_token_count = llava_model.get_num_tokens(line_text)

                wikimed_dataset_metadata.append({
                    "id": int(line_id),
                    "title": line_title,
                    "word_count": len(line_text),
                    "sentence_count": len(line_text.split('.')),
                    "model_token_count": model_token_count
                })
                progress_bar.update(len(line.encode('utf-8')))

    wikimed_dataset_metadata_df = pd.DataFrame(wikimed_dataset_metadata)
    print("+ WikiMed dataset metadata loaded.")
    return wikimed_dataset_metadata_df


def calculate_summary_statistics(column: pd.Series) -> dict[str, float]:
    return {
        "Min": column.min(),
        "Q1": column.quantile(0.25),
        "Median": column.quantile(0.5),
        "Q3": column.quantile(0.75),
        "Max": column.max()
    }


def get_dataset_row_by_doc_title(
    dataset_path: Path,
    dataset_metadata: pd.DataFrame,
    doc_title: int
) -> dict[str, Any]:
    try:
        row_index = dataset_metadata[dataset_metadata['title'] == doc_title].index.item()
    except Exception as e:
        raise ValueError(f"No document found with the title '{doc_title}'") from e

    row = json.loads(linecache.getline(filename=str(dataset_path), lineno=row_index + 1))
    return row
