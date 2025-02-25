from typing import Union
from langchain_core.messages import BaseMessage

from src.utils.enums import LogLevel
from src.utils.logger import LoggerManager


def log_conversation_messages(logger_manager: LoggerManager, messages: list[BaseMessage]) -> None:
    for message in messages:
        content_log_messages = []
        if isinstance(message.content, list):
            for sub_message in message.content:
                content_log_messages.append(__process_sub_message(sub_message))
        elif isinstance(message.content, str):
            content_log_messages.append(message.content)

        message_type = message.type.upper()
        log_message_elements = [
            f"---- Start of {message_type} message ----",
            *content_log_messages,
            f"---- End of {message_type} message ----"
        ]
        logger_manager.log(level=LogLevel.INFO, message="\n\n".join(log_message_elements))


# ====================
# Private Functions
# ====================


def __process_sub_message(sub_message: Union[str, dict]) -> list[str]:
    if isinstance(sub_message, str):
        return sub_message.content

    if isinstance(sub_message, dict):
        if sub_message["type"] == "text":
            return sub_message["text"]

        if sub_message["type"] == "image_url":
            base_64_image = f"Base64 Image: {sub_message["image_url"][:100]} ..."
            return base_64_image

    return None
