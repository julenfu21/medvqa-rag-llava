from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.utils.enums import VQAStrategyType
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


class ZeroShotVQAStrategy(BaseVQAStrategy):

    @property
    def strategy_type(self) -> VQAStrategyType:
        return VQAStrategyType.ZERO_SHOT


    def _init_strategy(self) -> None:
        pass


    def load_ollama_model(self, model_name: str) -> BaseChatModel:

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
                        {
                            "type": "text",
                            "text": question
                        },
                    ]
                ),
            ]

        llm = ChatOllama(model=model_name, temperature=0, num_predict=1)
        chain = prompt_template | llm | StrOutputParser()
        return chain


    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        base64_image: str
    ) -> str:
        possible_answers = " ".join(
            [f"{letter} - {answer}" for letter, answer in possible_answers.items()]
        )
        question_with_possible_answers = f"{question} {possible_answers}"

        output = model.invoke({
            "question": question_with_possible_answers,
            "image": base64_image
        })
        return output.strip()
