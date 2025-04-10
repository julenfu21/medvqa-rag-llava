from typing import Optional

import ipywidgets as widgets

from src.ui.widgets.output_widget_manager import OutputWidgetManager
from src.utils.data_definitions import ModelAnswerResult


def visualize_qa_pair_row(
    output_widget_manager: OutputWidgetManager,
    display_image: bool,
    row: dict,
    model_answer_result: Optional[ModelAnswerResult] = None,
    possible_options: Optional[list[str]] = None
) -> None:
    output_widget_manager.clear_content()
    if possible_options is None:
        possible_options = ['A', 'B', 'C', 'D']

    # Display row id
    output_widget_manager.display_text_content(
        content=str(row['index']),
        title="Question ID"
    )

    # Display question
    output_widget_manager.display_text_content(
        content=row['question'],
        extra_css_style="margin-bottom: 20px;",
        title="Question"
    )

    # Display context image
    if display_image:
        output_widget_manager.display_text_content(
            content="",
            title="Context Image"
        )
        output_widget_manager.display_base64_image(base64_image=row['image'])

    # Display possible options and model answer (if provided)
    __display_possible_options(
        output_widget_manager=output_widget_manager,
        row=row,
        possible_options=possible_options,
        model_answer=model_answer_result.answer if model_answer_result else None
    )
    if model_answer_result:
        model_answer = model_answer_result.answer
        css_style = "margin-bottom: 20px 0;"

        if model_answer not in possible_options:
            model_answer = f"{model_answer} (Invalid model answer)"
            css_style = f"{css_style} color: rgb(184, 134, 11)"

        output_widget_manager.display_text_content(
            content=model_answer,
            extra_css_style=css_style,
            title="Model Answer"
        )

def __display_possible_options(
    output_widget_manager: OutputWidgetManager,
    row: dict,
    possible_options: list[str],
    model_answer: Optional[str] = None
) -> None:
    formatted_options = []

    for option in possible_options:
        color = None
        bold = False

        if option == row['correct_option']:
            color = (
                'rgb(0, 0, 255)' 
                if model_answer and model_answer not in possible_options
                else 'rgb(0, 255, 0)'
            )
            bold = True
        elif option == model_answer:
            color = 'rgb(255, 0, 0)'
            bold = True

        formatted_options.append(
            __format_option(
                option_letter=option,
                option_sentence=row[option],
                color=color,
                bold=bold
            )
        )

    formatted_options_html = "".join(formatted_options)
    output_widget_manager.display_text_content(
        content=formatted_options_html,
        extra_css_style="margin-bottom: 20px;",
        title="Possible Answers"
    )

def __format_option(
    option_letter: str,
    option_sentence: str,
    color: str = "rgb(0, 0, 0)",
    bold: bool = False
) -> str:
    style = f"color: {color}; margin: 0px; padding: 0px;"
    bold_tag = "<b>" if bold else ""
    bold_end_tag = "</b>" if bold else ""

    return (
        f"<p style='{style}'>"
        f"{bold_tag}{option_letter}) {option_sentence}{bold_end_tag}"
        "</p>"
    )


def visualize_options_subset(
    output_widget_manager: OutputWidgetManager,
    options_widgets: list[widgets.Widget]
) -> None:
    widget_types_map = {
        widgets.Checkbox: lambda w: f"- {w.description}: {'✅' if w.value else '❌'}",
        widgets.Dropdown: lambda w: f"- {w.description} {w.label}",
        widgets.BoundedIntText: lambda w: f"- {w.description} {w.value}"
    }
    option_subset_rows = []

    for widget in options_widgets:
        if widget.disabled:
            continue

        widget_type = type(widget)
        try:
            option_subset_rows.append(widget_types_map[widget_type](widget))
        except KeyError as e:
            raise ValueError(f"Unexpected widget type: {widget_type}") from e

    output_widget_manager.display_text_content(
        content="\n".join(option_subset_rows),
        extra_css_style="margin-left: 60px;"
    )
