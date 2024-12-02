import json
from pathlib import Path

from datasets import Dataset
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from tqdm import tqdm


class RAGQVQAModel:

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
            relevant_docs = data["relevant_docs"]

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
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "text",
                            "text": relevant_docs
                        }
                    ]
                ),
            ]

        llm = ChatOllama(model=model_name, temperature=0, num_predict=1)
        chain = prompt_template | llm | StrOutputParser()
        return chain


    def index_wikimed_data(
        self,
        index_dir: Path,
        index_name: str,
        embedding_model_name: str,
        relevant_docs_count: int
    ) -> VectorStore:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            # model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        )
        print("Embeddings loaded!")

        # Load FAISS index
        index = FAISS.load_local(
            folder_path=index_dir,
            index_name=index_name,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("Index loaded!")

        # Load retriever from index
        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": relevant_docs_count}
        )
        print("Retriever loaded!")
        return retriever


    def __save_evaluation_results(self, data: dict, save_path: Path) -> None:
        Path.mkdir(save_path, exist_ok=True)
        results_filename = f"{self.__country}_{self.__file_type}_RAG_V1_evaluation.json"
        file_path = save_path / results_filename

        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


    def generate_answer_from_row(
        self, row: dict, possible_options: list[str], retriever: BaseRetriever
    ) -> str:
        def format_docs(docs) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        possible_answers = " ".join(
            [f"({option} - {row[option]})" for option in possible_options]
        )
        question = f"{row['question']} {possible_answers}"

        output = self.__model.invoke(
            {
                "question": question,
                "image": row["image"],
                "relevant_docs": format_docs(retriever.invoke(question)),
            }
        )
        return output.strip()


    def evaluate(
        self, dataset: Dataset, retriever: BaseRetriever, save_path: Path
    ) -> None:
        gold_options = {}
        predicted_options = {}
        possible_options = ["A", "B", "C", "D"]

        for row in tqdm(
            dataset,
            desc=f"Evaluating model ({self.__country}_{self.__file_type} subset) ...",
        ):
            row_index = row["index"]
            gold_options[row_index] = row["correct_option"]
            predicted_options[row_index] = self.generate_answer_from_row(
                row, possible_options, retriever
            )

        accuracy = [
            gold == prediction
            for gold, prediction in zip(
                gold_options.values(), predicted_options.values()
            )
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
        )
