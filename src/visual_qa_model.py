import json
from pathlib import Path

from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

from src.utils.data_definitions import ModelAnswerResult
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


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
        print(
            f"- Loading {capitalized_model_name} model (prompt template: "
            f"{self.__visual_qa_strategy.prompt_type}) ..."
        )
        ollama_model = self.__visual_qa_strategy.load_ollama_model(self.__model_name)
        print(
            f"+ {capitalized_model_name} model "
            f"(prompt template: {self.__visual_qa_strategy.prompt_type}) loaded.")
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
    ) -> ModelAnswerResult:
        if verbose:
            print(f"- Generating Answer for Question (ID: {row['index']}) ...")

        model_answer_result = self.__visual_qa_strategy.generate_answer_from_row(
            model=self.__model,
            question=row['question'],
            possible_answers={option: row[option] for option in possible_options},
            base64_image=row['image']
        )

        if verbose:
            print(f"+ Answer for Question (ID: {row['index']}) generated.")

        return model_answer_result


    def __is_answer_well_formatted(
        self,
        answer: str,
        possible_options: list[str]
    ) -> bool:
        return answer in possible_options


    def __compute_evaluation_metrics(
        self,
        gold_options: dict[str, str],
        predicted_options: dict[str, str],
        possible_options: list[str],
        relevant_documents: dict
    ) -> dict:
        accuracy = [
            gold == prediction
            for gold, prediction in zip(gold_options.values(), predicted_options.values())
        ].count(True) / len(gold_options)

        predictions = {}
        well_formatted_count = 0
        total_docs_used = 0
        total_mean_docs_length = 0

        for (id_gold, gold_option), pred_option, relevant_docs in zip(
            gold_options.items(),
            predicted_options.values(),
            relevant_documents.values()
        ):
            is_answer_well_formatted = self.__is_answer_well_formatted(
                answer=pred_option, possible_options=possible_options
            )
            well_formatted_count += 1 if is_answer_well_formatted else 0

            predictions[id_gold] = {
                "predicted_answer": pred_option,
                "gold_answer": gold_option,
                "is_well_formatted": is_answer_well_formatted,
            }
            if relevant_docs:
                mean_docs_length = sum(doc["length"] for doc in relevant_docs) / len(relevant_docs)
                total_mean_docs_length += mean_docs_length
                total_docs_used += len(relevant_docs)

                predictions[id_gold].update({
                    "relevant_docs": relevant_docs,
                    "mean_relevant_docs_length": mean_docs_length
                })

        return {
            "accuracy": accuracy,
            "percentage_well_formatted": well_formatted_count / len(predictions),
            **(
                {
                    "mean_relevant_docs_used": total_docs_used / len(predictions),
                    "mean_relevant_docs_length": total_mean_docs_length / len(predictions),
                } if total_docs_used != 0 else {}
            ),
            "predictions": predictions
        }


    def __save_evaluation_results(
        self,
        data: dict,
        save_path: Path,
        results_filename: str
    ) -> None:
        strategy_name = self.__visual_qa_strategy.strategy_type.value
        save_path = save_path / strategy_name
        save_path.mkdir(parents=True, exist_ok=True)

        results_filepath = save_path / results_filename
        with open(results_filepath, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


    def __generate_results_filename(self) -> str:
        prompt_type_name = self.__visual_qa_strategy.prompt_type.value
        return f'{self.__country}_{self.__file_type}_{prompt_type_name}_evaluation.json'


    def evaluate(self, dataset: Dataset, save_path: Path) -> None:
        possible_options = ["A", "B", "C", "D"]
        gold_options = {}
        predicted_options = {}
        relevant_documents = {}

        # Obtain gold and predicted options, and relevant documents if any
        for row in tqdm(
            dataset,
            desc=f"- Evaluating model ({self.__country}_{self.__file_type} subset) ...",
        ):
            row_index = row["index"]
            gold_options[row_index] = row["correct_option"]

            model_answer_result = self.generate_answer_from_row(row, possible_options)
            predicted_options[row_index] = model_answer_result.answer
            current_relevant_documents = []
            for document in model_answer_result.relevant_documents:
                current_relevant_documents.append({
                    "doc_title": document.metadata['title'],
                    "length": len(document.page_content)
                })
            relevant_documents[row_index] = current_relevant_documents

        evaluation_metrics = self.__compute_evaluation_metrics(
            gold_options, predicted_options, possible_options, relevant_documents
        )

        self.__save_evaluation_results(
            data=evaluation_metrics,
            save_path=save_path,
            results_filename=self.__generate_results_filename()
        )
        print(f"+ Model evaluation ({self.__country}_{self.__file_type} subset) completed.")
