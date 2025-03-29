from typing import Optional

import ipywidgets as widgets
from IPython.display import display


class OutputWidgetManager:

    def __init__(self, initial_content: str) -> None:
        self.__initial_content = initial_content
        self.__output_widget = widgets.Output(
            layout=widgets.Layout(
                width="50%",
                overflow="hidden",
                margin="0px 20px 0px 0px"
            )
        )

        with self.__output_widget:
            self.display_text_content(content=self.__initial_content)

    @property
    def output_widget(self) -> widgets.Output:
        return self.__output_widget


    def reset_content(self) -> None:
        self.clear_content()

        with self.__output_widget:
            self.display_text_content(content=self.__initial_content)

    def clear_content(self) -> None:
        self.__output_widget.clear_output()

    def display_text_content(
        self,
        content: str,
        extra_css_style: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        formatted_content = content.strip().replace("\n", "<br>")
        inline_css_style = (
            "'white-space: normal; "
            "overflow-wrap: break-word; "
            "font-family: monospace; "
            "font-size: 14px;"
            f"{extra_css_style}'"
        )
        if title:
            formatted_content = f"<b>{title}:</b> {formatted_content}"

        html_content = f"""
        <div style={inline_css_style}>
            {formatted_content}
        </div>
        """

        with self.__output_widget:
            display(widgets.HTML(value=html_content))

    def display_base64_image(
        self,
        base64_image: str
    ) -> None:
        html_content = (
            f"<img src='data:image/png;base64,{base64_image}'"
            "style='width:100%; height:auto; margin-bottom: 20px;'>"
        )

        with self.__output_widget:
            display(widgets.HTML(value=html_content))
