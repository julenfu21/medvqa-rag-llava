from abc import ABC, abstractmethod

from PIL import Image
from langchain_core.language_models.chat_models import BaseChatModel


class VisualQAStrategy(ABC):

    @abstractmethod
    def load_ollama_model(self, model_name: str) -> BaseChatModel:
        pass

    @abstractmethod
    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        image: Image.Image
    ) -> str:
        pass

    @abstractmethod
    def generate_results_filename(self, country: str, file_type: str) -> str:
        pass
