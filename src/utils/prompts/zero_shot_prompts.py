from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.enums import ZeroShotPromptType
from src.utils.prompts.prompts_helpers import log_conversation_messages


def v1_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    logger_manager = data["logger_manager"]

    if image:
        system_message = (
            "You are an assistant that only responds with a single letter: A, B, C, or "
            "D. For each question, you should consider the provided options and the "
            "image, and answer with exactly one letter that best matches the correct "
            "choice. Answer with a single letter only, without any explanations or "
            "additional information."
        )
        human_message = [
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            },
            {
                "type": "text",
                "text": question
            }
        ]
    else:
        system_message = (
            "You are an assistant that only responds with a single letter: A, B, C, or "
            "D. For each question, you should consider the provided options "
            "and answer with exactly one letter that best matches the correct "
            "choice. Answer with a single letter only, without any explanations or "
            "additional information."
        )
        human_message = [
            {
                "type": "text",
                "text": question
            }
        ]

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


# Explicitly guide the model to focus on the image and the question
def v2_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    logger_manager = data["logger_manager"]

    if image:
        system_message = (
            "You are an assistant that only responds with a single letter: A, B, C, or D. "
            "For each question, you should carefully consider the image and the question. "
            "Focus on identifying the correct answer based solely on the image and the "
            "question. Ignore any irrelevant details or assumptions that are not directly "
            "visible in the image."
        )
        human_message = [
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            },
            {
                "type": "text",
                "text": question
            }
        ]
    else:
        system_message = (
            "You are an assistant that only responds with a single letter: A, B, C, or D. "
            "For each question, you should carefully consider the question. "
            "Focus on identifying the correct answer based solely on the "
            "question."
        )
        human_message = [
            {
                "type": "text",
                "text": question
            }
        ]

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


# Clarify the task +
# Ensure the model fully understands the importance of focusing on the image and question
def v3_prompt_template(data: dict) -> list:
    question = data["question"]
    image = data["image"]
    logger_manager = data["logger_manager"]

    if image:
        task_definition_system_message = (
            "You are an assistant that only responds with a single letter: A, B, C, or D. "
            "Your task is to carefully analyze the provided image and question."
        )
        task_details_system_message = (
            "Pay close attention to details in the image that relate to the question. "
            "Do not make assumptions or consider anything outside of what is visible in the "
            "image."
        )
        task_definition_messages = [
            task_definition_system_message,
            task_details_system_message
        ]
        human_message = [
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",
            },
            {
                "type": "text",
                "text": question
            }
        ]
    else:
        task_definition_system_message = (
            "You are an assistant that only responds with a single letter: A, B, C, or D. "
            "Your task is to carefully analyze the provided question."
        )
        task_definition_messages = [task_definition_system_message]
        human_message = [
            {
                "type": "text",
                "text": question
            }
        ]

    messages = [
        *[
            SystemMessage(content=message)
            for message in task_definition_messages
        ],
        SystemMessage(
            content=(
                "Answer with a single letter (A, B, C, or D) that best matches the correct choice. "
                "Provide no explanations or additional information."
            )
        ),
        HumanMessage(content=human_message)
    ]

    if logger_manager:
        log_conversation_messages(logger_manager=logger_manager, messages=messages)

    return messages


ZERO_SHOT_PROMPTS = {
    ZeroShotPromptType.V1: v1_prompt_template,
    ZeroShotPromptType.V2: v2_prompt_template,
    ZeroShotPromptType.V3: v3_prompt_template
}
