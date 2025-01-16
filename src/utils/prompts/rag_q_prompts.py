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


# Guide the model to prioritize specific parts of the documents that are more relevant
def v2_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]

    return [
        SystemMessage(
            content=(
                "You are an assistant that only responds with a single letter: A, B, C, or D. "
                "Focus on the most relevant sections of the documents that match the key terms "
                "or ideas from the question. Always consider the image and the options provided. "
                "Answer with a single letter only, without explanations or additional details. "
                "If the documents contain unrelated information, ignore it."
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


# Secondary system message to reinforce the priority of focusing on relevant information
def v3_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]

    return [
        # Defines the task broadly.
        SystemMessage(
            content=(
                "You are an assistant that only responds with a single letter: A, B, C, or D. "
                "For each question, consider the provided options, the image, and the documents."
            )
        ),
        # Specifically emphasize focusing on relevant sections of the documents
        SystemMessage(
            content=(
                "Focus on the sections of the documents that are most relevant to the question. "
                "Do not let irrelevant details distract you. Answer only based on key details "
                "that align with the question."
            )
        ),
        # Reinforce the required output format (a single letter)
        SystemMessage(
            content=(
                "Your answer must be a single letter: A, B, C, or D. Provide no explanations, "
                "and do not include any additional text."
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


# Document Wrapping and Explicit Instructions
def v4_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]

    return [
        SystemMessage(
            content=(
                "You are an assistant that only responds with a single letter: A, B, C, or D. "
                "For each question, carefully analyze the image and the question. Additionally, "
                "refer to the content inside the <rag_docs> tag for relevant details. "
                "Do not consider anything outside of the <rag_docs> tag. "
                "Answer with a single letter only, without explanations or additional information."
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
                    "text": f"<rag_docs>\n{relevant_docs}\n</rag_docs>"
                }
            ]
        ),
    ]




RAG_Q_PROMPTS = {
    RagQPromptType.V1: v1_prompt_template,
    RagQPromptType.V2: v2_prompt_template,
    RagQPromptType.V3: v3_prompt_template,
    RagQPromptType.V4: v4_prompt_template
}
