import json
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from tqdm import tqdm

from src.utils.data_definitions import DocSplitterOptions, ModelAnswerResult, VQAStrategyDetail
from src.utils.enums import DocumentSplitterType, VQAStrategyType
from src.utils.logger import LoggerManager
from src.utils.text_splitters.base_splitter import BaseSplitter
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
        verbose: bool = False,
        use_image: bool = True,
        logger_manager: Optional[LoggerManager] = None,
        **kwargs: dict[str, Any]
    ) -> ModelAnswerResult:
        if verbose:
            print(f"- Generating Answer for Question (ID: {row['index']}) ...")

        model_answer_result = self.__visual_qa_strategy.generate_answer_from_row(
            model=self.__model,
            question=row['question'],
            possible_answers={option: row[option] for option in possible_options},
            base64_image=row['image'] if use_image else None,
            logger_manager=logger_manager,
            **kwargs
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
        relevant_documents: dict,
        are_shortened_documents: bool
    ) -> dict:
        accuracy = [
            gold == prediction
            for gold, prediction in zip(gold_options.values(), predicted_options.values())
        ].count(True) / len(gold_options)

        predictions = {}
        well_formatted_count = 0
        total_docs_used = 0
        total_mean_docs_length = 0
        total_mean_original_docs_length = 0
        total_mean_shortened_docs_length = 0

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
                if are_shortened_documents:
                    total_shortened_docs_length = sum(
                        doc['shortened_doc_length'] for doc in relevant_docs
                    )
                    mean_shortened_docs_length = total_shortened_docs_length / len(relevant_docs)
                    total_mean_shortened_docs_length += mean_shortened_docs_length

                    total_original_docs_length = sum(
                        doc['original_doc_length'] for doc in relevant_docs
                    )
                    mean_original_docs_length = total_original_docs_length / len(relevant_docs)
                    total_mean_original_docs_length += mean_original_docs_length

                    predictions[id_gold].update({
                        "relevant_docs": relevant_docs,
                        "mean_original_docs_length": mean_original_docs_length,
                        "mean_shortened_docs_length": mean_shortened_docs_length
                    })
                else:
                    total_docs_length = sum(doc['doc_length'] for doc in relevant_docs)
                    mean_docs_length = total_docs_length / len(relevant_docs)
                    total_mean_docs_length += mean_docs_length

                    predictions[id_gold].update({
                        "relevant_docs": relevant_docs,
                        "mean_relevant_docs_length": mean_docs_length
                    })

                total_docs_used += len(relevant_docs)

        evaluation_metrics = {
            "accuracy": accuracy,
            "percentage_well_formatted": well_formatted_count / len(predictions)
        }

        if total_docs_used != 0:
            other_metrics = {
                "mean_relevant_docs_used": total_docs_used / len(predictions)
            }
            if are_shortened_documents:
                other_metrics.update({
                    "mean_original_docs_length": total_mean_original_docs_length / len(predictions),
                    "mean_shortened_docs_length": total_mean_shortened_docs_length / len(predictions)
                })
            else:
                other_metrics.update({
                    "mean_relevant_docs_length": total_mean_docs_length / len(predictions)
                })
            evaluation_metrics.update(other_metrics)

        evaluation_metrics["predictions"] = predictions

        return evaluation_metrics


    def __save_evaluation_results(
        self,
        data: dict,
        results_path: Path,
        use_image: bool,
        doc_splitter: Optional[BaseSplitter],
        should_apply_rag_to_question: Optional[bool]
    ) -> None:
        vqa_strategy_detail = VQAStrategyDetail(
            country=self.__country,
            file_type=self.__file_type,
            use_image=use_image,
            vqa_strategy_type=self.__visual_qa_strategy.strategy_type,
            prompt_type=self.__visual_qa_strategy.prompt_type,
            relevant_docs_count=(
                self.__visual_qa_strategy.relevant_docs_count
                if self.__visual_qa_strategy.strategy_type != VQAStrategyType.ZERO_SHOT else None
            ),
            doc_splitter_options=DocSplitterOptions(
                doc_splitter_type=doc_splitter.document_splitter_type,
                token_count=doc_splitter.token_count,
                add_title=doc_splitter.add_title,
                **(
                    {
                        "chunk_size": doc_splitter.chunk_size,
                        "chunk_overlap": doc_splitter.chunk_overlap
                    }
                    if doc_splitter.document_splitter_type == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER
                    else {}
                )
            ) if doc_splitter else None,
            should_apply_rag_to_question=(
                should_apply_rag_to_question
                if self.__visual_qa_strategy.strategy_type == VQAStrategyType.RAG_Q_AS else None
            )
        )
        full_results_filepath = vqa_strategy_detail.generate_evaluation_results_filepath(
            evaluation_results_folder=results_path
        )

        full_results_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(full_results_filepath, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


    def evaluate(
        self,
        dataset: Dataset,
        results_path: Path,
        use_image: bool,
        **kwargs: dict[str, Any]
    ) -> None:
        possible_options = ["A", "B", "C", "D"]
        doc_splitter = kwargs.get("doc_splitter")
        should_apply_rag_to_question = kwargs.get("should_apply_rag_to_question")
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
            model_answer_result = self.generate_answer_from_row(
                row, possible_options, use_image, **kwargs
            )
            predicted_options[row_index] = model_answer_result.answer

            current_relevant_documents = []
            are_shortened_documents = False
            if model_answer_result.shortened_relevant_documents:
                are_shortened_documents = True
                for original_doc, shortened_doc in zip(
                    model_answer_result.original_relevant_documents,
                    model_answer_result.shortened_relevant_documents
                ):
                    current_relevant_documents.append({
                        "doc_title": original_doc.metadata['title'],
                        "original_doc_length": len(original_doc.page_content),
                        "shortened_doc_length": len(shortened_doc),
                        "shortened_doc_content": shortened_doc
                    })
            else:
                for document in model_answer_result.original_relevant_documents:
                    current_relevant_documents.append({
                        "doc_title": document.metadata['title'],
                        "doc_length": len(document.page_content)
                    })
            relevant_documents[row_index] = current_relevant_documents

        evaluation_metrics = self.__compute_evaluation_metrics(
            gold_options,
            predicted_options,
            possible_options,
            relevant_documents,
            are_shortened_documents
        )

        self.__save_evaluation_results(
            data=evaluation_metrics,
            results_path=results_path,
            use_image=use_image,
            doc_splitter=doc_splitter,
            should_apply_rag_to_question=should_apply_rag_to_question
        )
        print(f"+ Model evaluation ({self.__country}_{self.__file_type} subset) completed.")
