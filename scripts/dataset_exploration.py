import argparse
import json
from pathlib import Path

from datasets import Dataset, disable_progress_bars

from utils.dataset_helpers import get_dataset_row_by_id, load_vqa_dataset


def get_qa_pair_in_json(dataset: Dataset, question_id: int) -> None:
    row = get_dataset_row_by_id(dataset=dataset, question_id=question_id)
    possible_options = ['A', 'B', 'C', 'D']

    qa_pair_data = {
        'question_id': question_id,
        'question': row['question'],
        'context_image': f"{row['image'][:300]} ...",
        'options': [
            {'label': option, 'text': row[option]}
            for option in possible_options
        ],
        'correct_option': row['correct_option']
    }

    print(json.dumps(qa_pair_data, indent=4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Script to visualize a QA pair in JSON format given a question ID."
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
    parser.add_argument("--question_id",
                        type=int, required=True,
                        help="ID of the question to be consulted in the dataset")
    return parser.parse_args()


def main() -> None:
    disable_progress_bars()
    args = parse_args()

    world_med_qa_v_dataset = load_vqa_dataset(
        data_path=args.dataset_dir,
        country=args.country,
        file_type=args.file_type
    )
    get_qa_pair_in_json(
        dataset=world_med_qa_v_dataset,
        question_id=args.question_id
    )


if __name__ == "__main__":
    main()
