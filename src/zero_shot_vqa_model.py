import json
from pathlib import Path

from datasets import Dataset, load_dataset
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from tqdm import tqdm


class ZeroShotVQAModel:

    def __init__(
        self, model_name: str, country: str = "spain", file_type: str = "english"
    ) -> None:
        self.__model = self.__load_ollama_model(model_name)
        self.__country = country
        self.__file_type = file_type


    def __load_ollama_model(self, model_name: str) -> BaseChatModel:
        def prompt_template(data: dict) -> list:
            question = data["question"]
            image = data["image"]

            return [
                SystemMessage(
                    content=(
                        "You are an assistant that only responds with a single letter: A, B, C, or "
                        "D. For each question, you should consider the provided options and the"
                        "image, and answer with exactly one letter that best matches the correct "
                        "choice. Answer with a single letter only, without any explanations or "
                        "additional information."
                    )
                ),
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image}",
                        },
                        {"type": "text", "text": question},
                    ]
                ),
            ]

        llm = ChatOllama(model=model_name, temperature=0, num_predict=1)
        chain = prompt_template | llm | StrOutputParser()
        return chain


    def __save_evaluation_results(self, data: dict, save_path: Path) -> None:
        Path.mkdir(save_path, exist_ok=True)
        results_filename = f"{self.__country}_{self.__file_type}_evaluation.json"
        file_path = save_path / results_filename

        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


    def generate_answer_from_row(self, row: dict, possible_options: list[str]) -> str:
        possible_answers = " ".join(
            [f"({option} - {row[option]})" for option in possible_options]
        )
        question = f"{row['question']} {possible_answers}"

        output = self.__model.invoke({"question": question, "image": row["image"]})
        return output.strip()


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
            save_path=save_path
        )

if __name__ == "__main__":
    DATASET_DIR = Path("data/WorldMedQA-V")
    COUNTRY = "spain"
    FILE_TYPE = "english"

    # Set dataset file path
    dataset_filename = f"{COUNTRY}_{FILE_TYPE}_processed.tsv"
    data_filepath = str(DATASET_DIR / dataset_filename)


    world_med_qa_v_dataset = load_dataset(
        "csv",
        data_files=[data_filepath],
        sep="\t"
    )['train']
    llava_model = ZeroShotVQAModel(
        model_name="llava",
        country=COUNTRY,
        file_type=FILE_TYPE
    )
    llava_model.evaluate(
        dataset=world_med_qa_v_dataset.take(5),
        save_path=Path('evaluation_results')
    )
