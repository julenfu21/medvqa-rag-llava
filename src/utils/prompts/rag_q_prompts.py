from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.enums import RagQPromptType
from src.utils.prompts.prompts_helpers import log_conversation_messages


def v1_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]
    logger_manager = data["logger_manager"]

    messages = [
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

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


# Guide the model to prioritize specific parts of the documents that are more relevant
def v2_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]
    logger_manager = data["logger_manager"]

    messages = [
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

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


# Secondary system message to reinforce the priority of focusing on relevant information
def v3_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]
    logger_manager = data["logger_manager"]

    messages = [
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

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


# Document Wrapping and Explicit Instructions
def v4_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]
    logger_manager = data["logger_manager"]

    messages = [
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

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


def v5_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]
    logger_manager = data["logger_manager"]

    messages = [
        SystemMessage(
            content=(
                "You are an assistant that only responds with a single letter: A, B, C or D. "
                "For each question, you must consider the image and the possible options provided "
                "and answer with exactly one letter corresponding to the option that best matches "
                "the correct choice. Additionally, you must also leverage the context below to "
                "provide a suitable answer."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Context:\n\n{relevant_docs}"
                }
            ]
        ),
        SystemMessage(
            content=(
                "Remember that your answer must be just a single letter: A, B, C, or D. "
                "Do not provide any explanations, nor include any additional text."
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
                    "text": f"Question:\n\n{question}"
                }
            ]
        )
    ]

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


def v6_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    relevant_docs = data["relevant_docs"]
    logger_manager = data["logger_manager"]

    messages = [
        SystemMessage(
            content=(
                "You are an AI assistant that answers multiple-choice questions by selecting "
                "exactly one letter: A, B, C, or D. Each question consists of BOTH an image and "
                "a text-based question. Carefully analyze them together to determine the correct "
                "answer. Additionally, relevant context is provided below inside <rag_docs>. "
                "This context should only be used if you cannot determine the answer directly "
                "from the image and question. Do not use any external knowledge beyond what is "
                "explicitly provided."
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"<rag_docs>\n\n{relevant_docs}\n\n</rag_docs>"
                }
            ]
        ),
        SystemMessage(
            content=(
                "Your response must be strictly one of the following: A, B, C, or D. "
                "Do NOT provide explanations, additional text, punctuation, or extra characters. "
                "Your answer must be a single uppercase letter without any formatting."
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
                    "text": f"Question:\n\n{question}"
                }
            ]
        )
    ]

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


RAG_Q_PROMPTS = {
    RagQPromptType.V1: v1_prompt_template,
    RagQPromptType.V2: v2_prompt_template,
    RagQPromptType.V3: v3_prompt_template,
    RagQPromptType.V4: v4_prompt_template,
    RagQPromptType.V5: v5_prompt_template,
    RagQPromptType.V6: v6_prompt_template
}
