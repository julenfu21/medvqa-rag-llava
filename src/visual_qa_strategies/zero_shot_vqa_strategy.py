from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from src.utils.data_definitions import ArgumentSpec, ModelAnswerResult
from src.utils.enums import VQAStrategyType, ZeroShotPromptType
from src.utils.prompts.zero_shot_prompts import ZERO_SHOT_PROMPTS
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


class ZeroShotVQAStrategy(BaseVQAStrategy):

    @property
    def strategy_type(self) -> VQAStrategyType:
        return VQAStrategyType.ZERO_SHOT


    def _set_prompt_template(self) -> None:
        super()._validate_arguments(
            required_arguments=[
                ArgumentSpec(
                    name="prompt_type", expected_type=ZeroShotPromptType, value=self._prompt_type
                )
            ]
        )

        self._prompt_template = ZERO_SHOT_PROMPTS[self._prompt_type]


    def _init_strategy(
        self,
        **kwargs: dict[str, Any]
    ) -> None:
        arguments = []
        super()._validate_arguments(arguments, **kwargs)


    def load_ollama_model(self, model_name: str) -> BaseChatModel:
        llm = ChatOllama(model=model_name, temperature=0, num_predict=1)
        chain = self._prompt_template | llm | StrOutputParser()
        return chain


    def generate_answer_from_row(
        self,
        model: BaseChatModel,
        question: str,
        possible_answers: dict[str, str],
        base64_image: str,
    ) -> ModelAnswerResult:
        possible_answers = " ".join(
            [f"{letter} - {answer}" for letter, answer in possible_answers.items()]
        )
        question_with_possible_answers = f"{question} {possible_answers}"

        output = model.invoke({
            "question": question_with_possible_answers,
            "image": base64_image
        })
        return ModelAnswerResult(answer=output.strip())
