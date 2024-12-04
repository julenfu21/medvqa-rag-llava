import json
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

from visual_qa_strategies.visual_qa_strategy import VisualQAStrategy


class VisualQAModel:

    def __init__(
        self,
        strategy: VisualQAStrategy,
        model_name: str,
        country: str = "spain",
        file_type: str = "english"
    ) -> None:
        self.__visual_qa_strategy = strategy
        self.__model_name = model_name
        self.__model = self.__visual_qa_strategy.load_ollama_model(self.__model_name)
        self.__country = country
        self.__file_type = file_type


    @property
    def strategy(self) -> VisualQAStrategy:
        return self.__visual_qa_strategy


    @strategy.setter
    def strategy(self, visual_qa_strategy: VisualQAStrategy) -> None:
        self.__visual_qa_strategy = visual_qa_strategy
        self.__model = self.__visual_qa_strategy.load_ollama_model(self.__model_name)


    def generate_answer_from_row(self, row: dict, possible_options: list[str]) -> str:
        return self.__visual_qa_strategy.generate_answer_from_row(
            model=self.__model,
            question=row['question'],
            possible_answers={option: row[option] for option in possible_options},
            image=row['image']
        )


    def __save_evaluation_results(
        self,
        data: dict,
        save_path: Path,
        results_filename: str
    ) -> None:
        Path.mkdir(save_path, exist_ok=True)
        file_path = save_path / results_filename

        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


    def evaluate(self, dataset: Dataset, save_path: Path) -> None:
        gold_options = {}
        predicted_options = {}
        possible_options = ["A", "B", "C", "D"]

        for row in tqdm(
            dataset,
            desc=f"Evaluating model ({self.__country}_{self.__file_type} subset) ...",
        ):
            row_index = row["index"]
            gold_options[row_index] = row["correct_option"]
            predicted_options[row_index] = self.generate_answer_from_row(row, possible_options)

        accuracy = [
            gold == prediction
            for gold, prediction in zip(gold_options.values(), predicted_options.values())
        ].count(True) / len(gold_options)
        self.__save_evaluation_results(
            data={
                "accuracy": accuracy,
                "predictions": [
                    {
                        "question_id": id_gold,
                        "predicted_answer": pred_option,
                        "gold_answer": gold_option,
                    }
                    for (id_gold, gold_option), (id_pred, pred_option) in zip(
                        gold_options.items(), predicted_options.items()
                    )
                ],
            },
            save_path=save_path,
            results_filename=self.__visual_qa_strategy.generate_results_filename(
                country=self.__country,
                file_type=self.__file_type
            )
        )
