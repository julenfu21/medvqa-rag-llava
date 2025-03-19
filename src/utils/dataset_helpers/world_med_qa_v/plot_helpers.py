import base64
from collections import Counter
from io import BytesIO
from typing import Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from PIL import Image
from plotly.subplots import make_subplots

from src.utils.data_definitions import ModelAnswerResult
from src.utils.dataset_helpers.shared_plot_helpers import _display_formatted_section
from src.utils.enums import VQAStrategyType
from src.utils.string_formatting_helpers import prettify_strategy_name


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
    title: str,
    column_names: list[str] = None
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

    if column_names:
        if len(column_names) == len(evaluation_results):
            x_labels = column_names
        else:
            raise ValueError(
                f"'column_names' (len: {len(column_names)}) must have the same length as "
                f"'evaluation_results' (len: {len(evaluation_results)})"
            )
    else:
        x_labels = evaluation_results.index
    bar_chart = go.Figure(
        data=[
            go.Bar(
                name=column['name'],
                x=x_labels,
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
        "relevant_docs_count",
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
            "Relevant Documents Count: %{customdata[2]}<br>"
            "Document Splitter: %{customdata[3]}<br>"
            "Add Title: %{customdata[4]}<br>"
            "Token Count: %{customdata[5]}<br>"
            "Chunk Size: %{customdata[6]}<br>"
            "Chunk Overlap: %{customdata[7]}<br>"
        )
    )

    display(bar_chart)


def plot_rag_q_evaluation_results_by_groups(
    title: str,
    evaluation_results: pd.DataFrame,
    row_variable: str,
    column_variable: str,
    bar_graph_variable: str
) -> None:

    def prettify_variable_name(variable: str) -> str:
        return variable.replace('_', ' ').capitalize()

    column_name_to_short_str = {
        "relevant_docs_count": "rdc",
        "token_count": "tc",
        "prompt_type": "pt"
    }
    row_names = evaluation_results[row_variable].unique()
    column_names = evaluation_results[column_variable].unique()
    subplot_titles = [
        (
            f"{column_name_to_short_str[row_variable]}{row_name}_"
            f"{column_name_to_short_str[column_variable]}{column_name}"
        )
        for row_name in row_names
        for column_name in column_names
    ]
    rows = len(row_names)
    columns = len(column_names)
    evaluation_metrics_figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    for annotation in evaluation_metrics_figure.layout.annotations:
        annotation['yshift'] = 10

    columns_metadata = [
        {'name': 'Accuracy', 'data_column': 'accuracy'},
        {'name': 'Well-Formatted Answers', 'data_column': 'well_formatted_answers'}
    ]
    grouped_evaluation_results = evaluation_results.groupby([row_variable, column_variable])
    for row_index, row_name in enumerate(row_names):
        for column_index, column_name in enumerate(column_names):
            bar_graph_data = grouped_evaluation_results.get_group((row_name, column_name))
            for column in columns_metadata:
                evaluation_metrics_figure.add_trace(
                    trace=go.Bar(
                        x=bar_graph_data[bar_graph_variable],
                        y=bar_graph_data[column['data_column']],
                        name=subplot_titles[row_index * columns + column_index],
                        hovertemplate=(
                            column['name'] + ": %{y:.1%}<br>" +
                            prettify_variable_name(bar_graph_variable) + ": %{x}"
                        ),
                        marker={
                            'color': bar_graph_data[column['data_column']],
                            'colorscale': "Thermal",
                            'cmin': 0,
                            'cmax': 1,
                            'colorbar': {
                                'title': "Accuracy",
                                'tickvals': [i / 5 for i in range(6)],
                                'tickformat': ".0%",
                                'thickness': 20
                            }
                        }
                    ),
                    row=row_index + 1,
                    col=column_index + 1
                )

            evaluation_metrics_figure.update_xaxes(
                title_text=prettify_variable_name(bar_graph_variable),
                row=row_index + 1,
                col=column_index + 1
            )
            evaluation_metrics_figure.update_yaxes(
                title_text="Accuracy",
                row=row_index + 1,
                col=column_index + 1,
                range=[0, 1],
                tickvals=[i / 10 for i in range(11)],
                tickformat=".0%"
            )

    base_height_per_row = 400
    evaluation_metrics_figure.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'size': 22,
                'color': 'black',
                'family': 'Arial, sans-serif'
            },
            'pad': {'b': 30}
        },
        font={
            'family': "Arial, sans-serif",
            'size': 14,
            'color': "black"
        },
        showlegend=False,
        barmode="group",
        margin={'l': 50, 'r': 100, 't': 120, 'b': 50},
        width=1450,
        height=base_height_per_row * rows
    )

    display(evaluation_metrics_figure)


def display_evaluation_results_summary(
    evaluation_results_list: list[pd.DataFrame],
    separator_rows: Optional[list[int]] = None,
    highlighted_rows: Optional[list[int]] = None
) -> None:
    if separator_rows is None:
        separator_rows = []
    if highlighted_rows is None:
        highlighted_rows = []

    results_df = pd.concat(evaluation_results_list, ignore_index=True)
    results_df['vqa_strategy_type'] = results_df.apply(
        lambda row: _get_pretty_strategy_representation(
            row['vqa_strategy_type'], row['should_apply_rag_to_question']
        ),
        axis=1
    )
    results_df['add_title'] = results_df['add_title'].apply(_transform_add_title)
    column_mapping = {
        'country': 'Country',
        'file_type': 'File Type',
        'vqa_strategy_type': 'VQA Strategy',
        'prompt_type': 'Prompt',
        'relevant_docs_count': 'Relevant Docs. Count',
        'doc_splitter': 'Doc. Splitter',
        'add_title': 'Title',
        'token_count': 'Token Count',
        'accuracy': 'Accuracy',
        'well_formatted_answers': 'Well Formatted Answers'
    }
    results_df = results_df.rename(columns=column_mapping)[column_mapping.values()]

    header_style = {
        'selector': 'thead th', 
        'props': [
            ('background-color', '#007BFF'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('padding', '12px'),
            ('text-align', 'center')
        ]
    }
    odd_row_style = {
        'selector': 'tbody tr:nth-child(odd)',
        'props': [
            ('background-color', '#f2f2f2'),
            ('color', 'black')
        ]
    }
    even_row_style = {
        'selector': 'tbody tr:nth-child(even)',
        'props': [
            ('background-color', '#ffffff'),
            ('color', 'black')
        ]
    }
    padding_and_text_alignment = {
        'selector': 'tbody td', 
        'props': [
            ('padding', '10px'),
            ('text-align', 'center')
        ]
    }
    table_style = {
        'selector': 'table', 
        'props': [
            ('border-collapse', 'collapse'),
            ('width', '100%'),
            ('margin', '0 auto')
        ]
    }
    border_style = {
        'selector': 'th, td', 
        'props': [('border', '1px solid #ddd')]
    }
    separator_styles = [
        {
            "selector": f"tbody tr:nth-child({i})",
            "props": [("border-bottom", "3px solid black")]
        }
        for i in separator_rows
    ]
    highlight_styles = [
        {
            "selector": f"tbody tr:nth-child({i})",
            "props": [('background-color', 'yellow')]
        }
        for i in highlighted_rows
    ]


    table_styles = [
        header_style,
        odd_row_style,
        even_row_style,
        padding_and_text_alignment,
        table_style,
        border_style
    ] + separator_styles + highlight_styles

    styled_results_df = results_df.style.set_table_styles(table_styles).format({
        'Accuracy': '{:.4f}',
        'Well Formatted Answers': '{:.4f}'
    }).hide(axis='index')

    display(styled_results_df)


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


def _get_pretty_strategy_representation(
    vqa_strategy_name: str,
    should_apply_rag_to_question: Union[bool, str]
) -> str:
    pretty_strategy_name = prettify_strategy_name(vqa_strategy_name)

    if vqa_strategy_name == VQAStrategyType.RAG_Q_AS.value:
        if should_apply_rag_to_question in ('-', False):
            pretty_strategy_name += " (Answers Only)"
        else:
            pretty_strategy_name += " (Question and Answers)"

    return pretty_strategy_name


def _transform_add_title(add_title: Union[bool, str]) -> str:
    if add_title == '-':
        return add_title
    return "Yes" if add_title else "No"
