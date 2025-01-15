from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.enums import RagQPromptType


def v1_prompt_template(data: dict) -> list:
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


def v2_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]

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
                {
                    "type": "text",
                    "text": relevant_docs
                }
            ]
        ),
    ]


def v3_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]

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
                {
                    "type": "text",
                    "text": relevant_docs
                }
            ]
        ),
    ]


RAG_Q_PROMPTS = {
    RagQPromptType.V1: v1_prompt_template,
    RagQPromptType.V2: v2_prompt_template,
    RagQPromptType.V3: v3_prompt_template
}
