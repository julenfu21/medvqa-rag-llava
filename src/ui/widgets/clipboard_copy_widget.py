import re
import threading
import time

import ipywidgets as widgets
import pyperclip

from src.utils.enums import CommandType
from src.ui.widgets.widget_factory import create_button


class ClipboardCopyWidget:

    def __init__(
        self,
        header: str,
        command_type: CommandType,
        text_content: str
    ) -> None:
        self.__header = header
        self.__command_type = command_type
        self.__text_content = text_content

        # Main Layout Elements
        self.__root_widget = None
        self.__header_widget = None
        self.__content_html_widget = None
        self.__copy_button = None

        self.__create_layout()


    @property
    def root_widget(self) -> widgets.VBox:
        return self.__root_widget


    # ============================
    # Layout Creation Methods
    # ============================

    def __create_layout(self) -> None:
        self.__header_widget = widgets.HTML(
            value=(
                "<h1 style='text-align: center; margin-bottom: 15px;'>"
                f"{self.__header}"
                "</h1>"
            )
        )
        self.__content_html_widget = self.__create_content_html_widget()
        self.__copy_button = self.__create_copy_button()
        self.__copy_button.on_click(lambda _: self.__copy_html_widget_content())

        self.__root_widget = widgets.VBox(
            children=[
                self.__header_widget,
                widgets.HBox(
                    children=[
                        self.__content_html_widget,
                        self.__copy_button
                    ],
                    layout=widgets.Layout(
                        width="100%",
                        align_items="stretch",
                        overflow="visible",
                        margin="30px 0"
                    )
                )
            ],
            layout=widgets.Layout(
                width="100%",
                align_items="stretch",
                overflow="visible"
            )
        )

    def __create_content_html_widget(self) -> widgets.HTML:
        formatted_content = self.__text_content.strip().replace("\n", "<br>")
        inline_css_style = (
            "'white-space: normal; "
            "overflow-wrap: break-word; "
            "font-family: monospace; "
            "font-size: 14px;'"
        )
        return widgets.HTML(
            value=f"""
            <div style={inline_css_style}>
                {formatted_content}
            </div>""",
            layout=widgets.Layout(width="80%")
        )

    def __create_copy_button(self) -> widgets.Button:
        command_type_to_tooltip_message = {
            CommandType.LINUX_COMMAND: "Copy this Linux command to your clipboard",
            CommandType.PYTHON_CODE: "Copy this Python code snippet to your clipboard"
        }

        return create_button(
            description="Copy",
            icon="copy",
            tooltip=command_type_to_tooltip_message[self.__command_type],
            button_style="",
            width="100px"
        )

    def __copy_html_widget_content(self) -> None:
        if self.__command_type == CommandType.LINUX_COMMAND:
            raw_value = self.__content_html_widget.value.strip()
            second_line = raw_value.split('\n')[1].strip()
            raw_text = second_line.replace('<br>', '\n')
            formatted_raw_text = " ".join(
                line.lstrip('> ').strip()
                for line in raw_text.split('\\\n')
            )
            pyperclip.copy(formatted_raw_text)

        elif self.__command_type == CommandType.PYTHON_CODE:
            raw_text = self.__content_html_widget.value.strip().split('\n')[1]
            formatted_raw_text = "\n".join(line.strip() for line in raw_text.split('<br>'))
            no_span_tags_text = re.sub(
                pattern=r'<span style=["\']margin-left:\s(\d+)px;["\']>(.*?)</span>',
                repl=self.__replace_span,
                string=formatted_raw_text
            )
            pyperclip.copy(no_span_tags_text + '\n')

        self.__copy_button.icon = "check"
        self.__copy_button.description = "Copied!"
        self.__copy_button.disabled = True
        self.__copy_button.button_style = "success"
        threading.Thread(target=self.__reset_button).start()

    @staticmethod
    def __replace_span(match: re.Match) -> str:
        margin = match.group(1)
        content = match.group(2)
        tabs = '\t' * (int(margin) // 30)
        return tabs + content

    def __reset_button(self) -> None:
        time.sleep(2)
        self.__copy_button.icon = "copy"
        self.__copy_button.description = "Copy"
        self.__copy_button.disabled = False
        self.__copy_button.button_style = ""
