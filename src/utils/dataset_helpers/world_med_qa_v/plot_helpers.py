import base64
from collections import Counter
from io import BytesIO
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from PIL import Image

from src.utils.data_definitions import ModelAnswerResult
from src.utils.dataset_helpers.shared_plot_helpers import _display_formatted_section


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
    model_answer: ModelAnswerResult = None
) -> None:
    # Display row id
    _display_formatted_section(
        section_name="ID",
        section_style="margin: 20px 0;",
        section_content=str(row['index'])
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
        elif model_answer and option == model_answer.answer:
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


def display_bar_chart_on_evaluation_results(
    evaluation_results: pd.DataFrame,
    title: str
) -> None:
    columns_metadata = [
        {
            'name': 'Accuracy',
            'data_column': 'accuracy',
            'color': 'royalblue'
        },
        {
            'name': 'Well-Formatted Answers',
            'data_column': 'well_formatted_answers', 
            'color': 'darkorange'
        }
    ]
    bar_chart = go.Figure(
        data=[
            go.Bar(
                name=column['name'],
                x=evaluation_results.index,
                y=evaluation_results[column['data_column']],
                marker={
                    'color': column['color'],
                    'opacity': 0.8
                },
                width=0.4,
                text=evaluation_results[column['data_column']],
                texttemplate="%{y:.1%}",
                textposition="outside",
                cliponaxis=False
            )
            for column in columns_metadata
        ]
    )

    bar_chart.update_layout(
        barmode='group',
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'size': 22,
                'color': 'black',
                'family': 'Arial, sans-serif'
            }
        },
        xaxis_title="Model Evaluations",
        yaxis={
            'title': 'Accuracy',
            'tickvals': [i / 100 for i in range(0, 110, 10)],
            'ticktext': [f"{i}%" for i in range(0, 110, 10)],
            'range': [0, 1.1],
        },
        font={
            'family': "Arial, sans-serif",
            'size': 14,
            'color': "black"
        },
        barcornerradius=15,
        height=500
    )

    hover_columns = [
        "vqa_strategy_type",
        "prompt_type",
        "doc_splitter",
        "add_title",
        "token_count",
        "chunk_size",
        "chunk_overlap"
    ]
    bar_chart.update_traces(
        customdata=evaluation_results[hover_columns].values,
        hovertemplate=(
            "VQA Strategy Type: %{customdata[0]}<br>"
            "Prompt Type: %{customdata[1]}<br>"
            "Document Splitter: %{customdata[2]}<br>"
            "Add Title: %{customdata[3]}<br>"
            "Token Count: %{customdata[4]}<br>"
            "Chunk Size: %{customdata[5]}<br>"
            "Chunk Overlap: %{customdata[6]}<br>"
        )
    )

    display(bar_chart)


# ====================
# Private Functions
# ====================


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
