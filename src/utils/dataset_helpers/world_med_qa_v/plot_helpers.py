import base64
from collections import Counter
from io import BytesIO
from typing import Optional

import pandas as pd
import plotly.express as px
from IPython.display import HTML, display
from PIL import Image

from src.utils.data_definitions import ModelAnswerResult


def display_pie_chart_on_correct_answer_distribution(
    data_frame: pd.DataFrame,
    title: str,
) -> None:
    correct_answer_distribution = Counter(data_frame['correct_option'])
    correct_answer_distribution_df = pd.DataFrame({
        "correct_option": correct_answer_distribution.keys(),
        "count": correct_answer_distribution.values()
    })
    correct_answer_distribution_df = correct_answer_distribution_df.sort_values('correct_option')

    correct_answer_distribution_pie_chart = px.pie(
        data_frame=correct_answer_distribution_df,
        names='correct_option',
        values="count",
        title=title,
        hole=0.45,
        category_orders={
            "correct_option": sorted(correct_answer_distribution.keys())
        },
        color='correct_option',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    correct_answer_distribution_pie_chart.update_traces(
        textposition='inside',
        textinfo='percent+label+value',
        pull=[0.05] * len(correct_answer_distribution_df),
        textfont={
            "size": 18,
            "color": 'black',
            "weight": 'bold'
        }
    )

    correct_answer_distribution_pie_chart.update_layout(
        legend={
            'title': 'Possible Answers',
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5,
            'font': {'size': 14}
        },
        width=850,
        height=650,
        title={
            'x': 0.5,
            'font': {
                'size': 24,
                'color': "black"
            }
        },
    )

    display(correct_answer_distribution_pie_chart)


def visualize_qa_pair_row(
    row: dict,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    model_answer: ModelAnswerResult = None,
) -> None:
    # Display row id
    _display_formatted_section(
        section_name="ID",
        section_style="margin: 20px 0;",
        section_content=row['index']
    )

    # Display question
    _display_formatted_section(
        section_name="Question",
        section_style="margin-bottom: 20px;",
        section_content=row['question']
    )

    # Display context image
    _display_formatted_section(
        section_name="Context Image",
        section_style="margin-bottom: 20px;",
        section_content=""
    )
    _display_base64_image(
        base64_image=row['image'],
        width=image_width,
        height=image_height
    )

    # Display possible answers marking the gold (and the predicted) option
    formatted_options = []
    possible_options = ['A', 'B', 'C', 'D']
    for option in possible_options:
        if option == row['correct_option']:
            formatted_options.append(
                f"<p style='color: rgb(0, 255, 0);'><b>{option}) {row[option]}</b>"
            )
        elif option == model_answer.answer:
            formatted_options.append(
                f"<p style='color: rgb(255, 0, 0);'><b>{option}) {row[option]}</b>"
            )
        else:
            formatted_options.append(f"<p>{option}) {row[option]}")
    answer = "<br><br>" + "<br>".join(formatted_options)

    _display_formatted_section(
        section_name="Possible Answers",
        section_style="margin-top: 30px;",
        section_content=answer
    )

    if model_answer:
        _display_formatted_section(
            section_name="Model Answer",
            section_style="margin: 30px 0;",
            section_content=model_answer.answer
        )


# ====================
# Private Functions
# ====================


def _display_formatted_section(
    section_name: str,
    section_style: str,
    section_content: str | int
) -> None:
    section_text = f"""
    <div style='{section_style}'>
        <b>{section_name}:</b> {section_content}
    </div>
    """
    display(HTML(section_text))


def _display_base64_image(
    base64_image: str,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> None:
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    resized_image = _resize_image(image, width, height)
    display(resized_image)


def _resize_image(
    image: Image.Image,
    width: Optional[int],
    height: Optional[int]
) -> Image.Image:
    if width or height:
        original_width, original_height = image.size

        if width and not height:
            height = int((width / original_width) * original_height)
        elif height and not width:
            width = int((height / original_height) * original_width)

        image = image.resize((width, height), Image.Resampling.LANCZOS)

    return image
