import json
from pathlib import Path

from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

from visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


class VisualQAModel:

    def __init__(
        self,
        visual_qa_strategy: BaseVQAStrategy,
        model_name: str,
        country: str = "spain",
        file_type: str = "english"
    ) -> None:
        self.__visual_qa_strategy = visual_qa_strategy
        self.__model_name = model_name
        self.__model = self.__load_ollama_model()
        self.__country = country
        self.__file_type = file_type


    def __load_ollama_model(self) -> BaseChatModel:
        capitalized_model_name = self.__model_name.capitalize()
        print(f"- Loading {capitalized_model_name} Model ...")
        ollama_model = self.__visual_qa_strategy.load_ollama_model(self.__model_name)
        print(f"+ {capitalized_model_name} Model Loaded.")
        return ollama_model


    @property
    def visual_qa_strategy(self) -> BaseVQAStrategy:
        return self.__visual_qa_strategy


    @visual_qa_strategy.setter
    def visual_qa_strategy(self, visual_qa_strategy: BaseVQAStrategy) -> None:
        self.__visual_qa_strategy = visual_qa_strategy
        self.__model = self.__load_ollama_model()


    def generate_answer_from_row(
        self,
        row: dict,
        possible_options: list[str],
        verbose: bool = False
    ) -> str:
        if verbose:
            print(f"- Generating Answer for Question (ID: {row['index']}) ...")

        model_answer = self.__visual_qa_strategy.generate_answer_from_row(
            model=self.__model,
            question=row['question'],
            possible_answers={option: row[option] for option in possible_options},
            base64_image=row['image']
        )

        if verbose:
            print(f"+ Answer for Question (ID: {row['index']}) generated.")

        return model_answer


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


    def __generate_results_filename(self) -> str:
        strategy_name = self.__visual_qa_strategy.strategy_type.value
        return f'{self.__country}_{self.__file_type}_{strategy_name}_evaluation.json'


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
                "predictions": {
                    id_gold: {
                        "predicted_answer": pred_option,
                        "gold_answer": gold_option,
                    }
                    for (id_gold, gold_option), (id_pred, pred_option) in zip(
                        gold_options.items(), predicted_options.items()
                    )
                },
            },
            save_path=save_path,
            results_filename=self.__generate_results_filename()
        )
