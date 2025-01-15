from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.enums import ZeroShotPromptType


def v1_prompt_template(data: dict) -> list:
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


def v2_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]

    return [
        SystemMessage(
            content=(
                "-"
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


def v3_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]

    return [
        SystemMessage(
            content=(
                "-"
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


ZERO_SHOT_PROMPTS = {
    ZeroShotPromptType.V1: v1_prompt_template,
    ZeroShotPromptType.V2: v2_prompt_template,
    ZeroShotPromptType.V3: v3_prompt_template
}
