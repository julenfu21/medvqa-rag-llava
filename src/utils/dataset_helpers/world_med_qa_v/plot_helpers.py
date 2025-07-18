from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import Dataset
from IPython.display import display
from langchain_ollama import ChatOllama
from plotly.subplots import make_subplots

from src.utils.enums import DocumentSplitterType, VQAStrategyType
from src.utils.string_formatting_helpers import prettify_strategy_name


def display_sample_distribution_across_languages(dataset: dict[str, Dataset]) -> None:
    english_subsets = _get_subsets_data_by_split_type(splits_data=dataset, split_type='english')
    languages_sample_distribution_df = pd.DataFrame({
        "split": [
            _get_formatted_subset_prefix(subset_name)
            for subset_name in english_subsets.keys()
        ],
        "instances": [subset.num_rows for subset in english_subsets.values()]
    })

    pie_chart = px.pie(
        data_frame=languages_sample_distribution_df,
        names='split',
        values='instances',
        title="Sample Distribution Across Languages (WorldMedQA-V Dataset)",
        hole=0.20,
        category_orders={"split": ["Brazil", "Israel", "Japan", "Spain"]},
        color='split',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    pie_chart.update_traces(
        textposition='inside',
        textinfo='percent+label+value',
        pull=[0.03] * len(languages_sample_distribution_df),
        textfont={
            "size": 18,
            "color": 'black',
            "weight": 'bold'
        }
    )
    pie_chart.update_layout(
        legend={
            'title': 'Splits',
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
        }
    )

    display(pie_chart)


def display_ground_truth_answer_distribution(dataset: dict[str, Dataset]) -> None:
    english_subsets = _get_subsets_data_by_split_type(splits_data=dataset, split_type='english')
    split_names = ["Brazil", "Israel", "Japan", "Spain"]
    ground_truth_answer_distribution = []

    for split_name in split_names:
        formatted_split_name = split_name.lower() + "_english"
        correct_options = english_subsets[formatted_split_name]['correct_option']
        subset_answer_distribution = Counter(correct_options)
        subset_answer_distribution_percentage = {
            key: value * 100 / sum(subset_answer_distribution.values())
            for key, value in subset_answer_distribution.items()
        }
        ground_truth_answer_distribution.append(subset_answer_distribution_percentage)

    bar_colors = px.colors.qualitative.Pastel
    grouped_bar_chart = go.Figure(
        data=[
            go.Bar(
                name=split_name,
                x=sorted(answer_distribution.keys()),
                y=list(dict(sorted(answer_distribution.items())).values()),
                marker_color=color,
                hovertemplate=(
                    '<b>Answer:</b> %{x}<br>'
                    '<b>Percentage:</b> %{y:.2f}%'
                )
            )
            for color, split_name, answer_distribution in zip(
                bar_colors, split_names, ground_truth_answer_distribution
            )
        ]
    )
    grouped_bar_chart.update_layout(
        barmode='group',
        legend={
            'title': 'Splits',
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.3,
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
            },
            'text': "Ground-Truth Answer Distribution (WorldMedQA-V Dataset)"
        },
        xaxis={
            'title': 'Possible Answer',
            'showline': True,
            'linewidth': 1.5,
            'linecolor': 'black',
            'mirror': True,
            'ticks': 'outside',
            'tickfont': {'size': 14},
            'titlefont': {'size': 16},
        },
        yaxis={
            'title': 'Percentage',
            'showline': True,
            'linewidth': 1.5,
            'linecolor': 'black',
            'mirror': True,
            'ticks': 'outside',
            'tickfont': {'size': 14},
            'titlefont': {'size': 16},
            'gridcolor': 'lightgray',
            'zeroline': False
        },
        shapes=[
            {
                'type': 'rect',
                'xref': 'paper',
                'x0': 0,
                'x1': 1,
                'yref': 'y',
                'y0': 20,
                'y1': 30,
                'fillcolor': 'rgba(255, 0, 0, 0.15)',
                'line': {'width': 0},
                'layer': 'below'
            },
            {
                'type': 'line',
                'xref': 'paper',
                'x0': 0,
                'x1': 1,
                'yref': 'y',
                'y0': 25,
                'y1': 25,
                'line': {'color': 'red', 'width': 2, 'dash': 'dash'}
            }
        ]
    )
    grouped_bar_chart.update_yaxes(range=[0, 40], ticksuffix="%")

    display(grouped_bar_chart)


def display_image_overlap_across_languages(dataset: dict[str, Dataset]) -> None:
    english_subsets = _get_subsets_data_by_split_type(splits_data=dataset, split_type='english')
    image_overlap_values = []
    for (split_name1, split_data1), (split_name2, split_data2) in list(
        combinations(iterable=english_subsets.items(), r=2)
    ):
        formatted_split1_prefix = _get_formatted_subset_prefix(split_name1)
        formatted_split2_prefix = _get_formatted_subset_prefix(split_name2)
        image_overlap = len(list(set(split_data1['image']) & set(split_data2['image'])))
        image_overlap_values.append([
            formatted_split1_prefix, formatted_split2_prefix, image_overlap
        ])

    image_overlap_df = pd.DataFrame(
        image_overlap_values,
        columns=['split_1', 'split_2', 'shared_images']
    )
    mirrored_image_overlap_df = pd.concat([
        image_overlap_df,
        image_overlap_df.rename(columns={'split_1': 'split_2', 'split_2': 'split_1'})
    ])
    heatmap_data = mirrored_image_overlap_df.pivot(
        index='split_1', columns='split_2', values='shared_images'
    ).fillna(0).astype(int)

    custom_colorscale = [
        [0.0, '#c6dbef'],
        [0.2, '#9ecae1'],
        [0.4, '#6baed6'],
        [0.6, '#4292c6'],
        [0.8, '#2171b5'],
        [1.0, '#084594']
    ]
    heatmap = px.imshow(
        heatmap_data,
        text_auto=True,
        color_continuous_scale=custom_colorscale,
        labels={
            'x': 'Split 1',
            'y': 'Split 2',
            'color': 'Shared Images'
        },
        zmin=0,
        zmax=50,
        title="Image Overlap Across Languages (WorldMedQA-V Dataset)"
    )
    heatmap.update_traces(
        textfont={'size': 24},
        zmin=0.01,
        zmax=50
    )
    heatmap.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        width=850,
        height=650,
        title={
            'x': 0.5,
            'font': {'size': 24, 'color': "black"}
        },
        xaxis={
            'tickfont': {'size': 18},
            'side': 'top',
            'automargin': True
        },
        yaxis={
            'tickfont': {'size': 18},
            'automargin': True
        },
        coloraxis_colorbar={
            'tickformat': 'd',
            'dtick': 5
        }
    )

    display(heatmap)


def display_average_question_length_across_languages(dataset: dict[str, Dataset]) -> None:
    llava_model = ChatOllama(model="llava", temperature=0, num_predict=1)
    average_question_lengths = defaultdict(float)
    for split_name, split_data in dataset.items():
        for question in split_data['question']:
            average_question_lengths[split_name] += llava_model.get_num_tokens(question)
        average_question_lengths[split_name] /= len(split_data['question'])

    english_splits_lengths = _get_subsets_data_by_split_type(
        splits_data=average_question_lengths,
        split_type='english'
    )
    formatted_english_split_lengths = {
        _get_formatted_subset_prefix(split_name).lower(): average_length
        for split_name, average_length in english_splits_lengths.items()
    }
    local_language_split_lengths = _get_subsets_data_by_split_type(
        splits_data=average_question_lengths,
        split_type='local'
    )
    formatted_local_language_split_lengths = {
        _get_formatted_subset_prefix(split_name).lower(): average_length
        for split_name, average_length in local_language_split_lengths.items()
    }

    bar_colors = px.colors.qualitative.Antique
    split_names = ["brazil", "israel", "japan", "spain"]
    grouped_bar_chart = go.Figure(
        data=[
            go.Bar(
                name='Original Language',
                x=[split_name.capitalize() for split_name in split_names],
                y=[
                    formatted_local_language_split_lengths[split_name]
                    for split_name in split_names
                ],
                marker_color=bar_colors[0],
                hovertemplate=(
                    '<b>Split Name:</b> %{x}<br>'
                    '<b>Avg. Question Length:</b> %{y:.2f}'
                )
            ),
            go.Bar(
                name='English Translation',
                x=[split_name.capitalize() for split_name in split_names],
                y=[
                    formatted_english_split_lengths[split_name]
                    for split_name in split_names
                ],
                marker_color=bar_colors[1],
                hovertemplate=(
                    '<b>Split Name:</b> %{x}<br>'
                    '<b>Avg. Question Length:</b> %{y:.2f}'
                )
            )
        ]
    )
    grouped_bar_chart.update_layout(
        barmode='group',
        legend={
            'title': 'Subset Type',
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.3,
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
            },
            'text': "Average Question Length Across Languages (WorldMedQA-V Dataset)"
        },
        xaxis={
            'title': 'Splits',
            'showline': True,
            'linewidth': 1.5,
            'linecolor': 'black',
            'mirror': True,
            'ticks': 'outside',
            'tickfont': {'size': 14},
            'titlefont': {'size': 16},
        },
        yaxis={
            'title': 'Average Question Length (in Tokens)',
            'showline': True,
            'linewidth': 1.5,
            'linecolor': 'black',
            'mirror': True,
            'ticks': 'outside',
            'tickfont': {'size': 14},
            'titlefont': {'size': 16},
            'gridcolor': 'lightgray',
            'zeroline': False
        }
    )

    display(grouped_bar_chart)


def display_bar_chart_on_evaluation_results(
    evaluation_results: pd.DataFrame,
    title: str,
    x_axis_title: str,
    y_axis_title: str,
    x_dataframe_column_name: str,
    y_dataframe_column_name: str
) -> None:
    figure = px.bar(
        data_frame=evaluation_results,
        x=x_dataframe_column_name,
        y=y_dataframe_column_name,
        text=evaluation_results["accuracy"].round(4),
        color=evaluation_results[x_dataframe_column_name].astype(str)
    )

    figure.update_traces(
        textposition="inside",
        insidetextanchor="start",
        textfont={"size": 14, "color": "white"},
        hovertemplate=(
            f'<b>{x_axis_title}: </b>' + '%{x}<br>'
            f'<b>{y_axis_title}: </b>' + '%{y:.2%}<extra></extra>'
        )
    )

    figure.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=20,
        xaxis_title="Prompt Type",
        yaxis_title="Accuracy",
        xaxis_title_font_size=16,
        xaxis={'tickfont': {'size': 12}},
        yaxis_title_font_size=16,
        yaxis={'tickfont': {'size': 12}},
        barcornerradius=15,
        showlegend=False,
        bargap=0.2,
        margin={"t": 100}
    )

    display(figure)


def display_bar_chart_on_splitter_evaluation_results(
    splitter_evaluation_results: pd.DataFrame,
    document_splitter_type: DocumentSplitterType,
    title: str
) -> None:
    mean_accuracy = splitter_evaluation_results['accuracy'].mean()

    x_column = _get_splitter_results_column_names(
        splitter_evaluation_results, document_splitter_type
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=x_column,
            y=splitter_evaluation_results['accuracy'],
            text=splitter_evaluation_results['accuracy'].round(4),
            textposition='inside',
            insidetextanchor='start',
            hovertemplate=(
                '<b>Splitter Configuration:</b> %{x}<br>'
                '<b>Accuracy:</b> %{y:.2%}<extra></extra>'
            )
        )
    )

    figure.add_hline(
        y=mean_accuracy,
        line_dash="dash",
        line_color="red",
    )

    figure.add_annotation(
        xref='paper',
        x=1,
        y=mean_accuracy,
        text=f"Mean: {mean_accuracy:.2%}",
        showarrow=False,
        font={'size': 12, 'color': 'red'},
        bgcolor='white',
        bordercolor='red',
        borderwidth=1,
        borderpad=4,
        yshift=12,
        opacity=0.6
    )
    figure.update_traces(
        textposition='inside',
        insidetextanchor='start',
        textfont={'size': 14, 'color': 'white'}
    )
    figure.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=20,
        xaxis_title='Splitter Configuration',
        yaxis_title='Accuracy',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        barcornerradius=15
    )

    display(figure)


def display_bar_chart_on_best_mean_accuracy_results(
    mean_accuracy_values: dict[str, float],
    title: str
) -> None:
    data_frame = pd.DataFrame({
        "Splitter": list(mean_accuracy_values.keys()),
        "Accuracy": list(mean_accuracy_values.values())
    })

    figure = px.bar(
        data_frame=data_frame,
        x="Splitter",
        y="Accuracy",
        text=data_frame["Accuracy"].round(4),
        color="Splitter"
    )

    figure.update_traces(
        textposition="inside",
        insidetextanchor="start",
        textfont={"size": 14, "color": "white"},
        hovertemplate=(
            '<b>Splitter:</b> %{x}<br>'
            '<b>Accuracy:</b> %{y:.2%}<extra></extra>'
        )
    )

    figure.update_layout(
        title=title,
        title_x=0.5,
        title_font_size=20,
        xaxis_title="Document Splitter Type",
        yaxis_title="Mean Accuracy",
        xaxis_title_font_size=16,
        xaxis={'tickfont': {'size': 12}},
        yaxis_title_font_size=16,
        yaxis={'tickfont': {'size': 12}},
        barcornerradius=15,
        showlegend=False,
        bargap=0.2,
        margin={"t": 100}
    )

    display(figure)


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
        "prompt_type": "pt",
        "chunk_size": "cs"
    }
    row_names = evaluation_results[row_variable].unique()
    column_names = evaluation_results[column_variable].unique()
    bar_graph_names = evaluation_results[bar_graph_variable].unique()
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
    small_bar_chart_columns = len(bar_graph_names)
    evaluation_metrics_figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=subplot_titles,
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    for annotation in evaluation_metrics_figure.layout.annotations:
        annotation['yshift'] = 10

    colors = px.colors.qualitative.Pastel
    color_list = [colors[i] for i in range(small_bar_chart_columns)]
    grouped_evaluation_results = evaluation_results.groupby([row_variable, column_variable])
    for row_index, row_name in enumerate(row_names):
        for column_index, column_name in enumerate(column_names):
            bar_graph_data = grouped_evaluation_results.get_group((row_name, column_name))
            evaluation_metrics_figure.add_trace(
                trace=go.Bar(
                    x=bar_graph_data[bar_graph_variable],
                    y=bar_graph_data['accuracy'],
                    name=subplot_titles[row_index * columns + column_index],
                    hovertemplate=(
                        "<b>Accuracy:</b> %{y:.1%}<br>" +
                        f"<b>{prettify_variable_name(bar_graph_variable)}: </b>" + "%{x}" +
                        "<extra></extra>"
                    ),
                    marker={'color': color_list}
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
    excluded_indexes: Optional[list] = None,
    include_green_highlight: bool = False
) -> None:
    if excluded_indexes is None:
        excluded_indexes = []

    results_df = pd.concat(evaluation_results_list, ignore_index=True)
    results_df[['country', 'file_type']] = results_df[['country', 'file_type']].apply(
        lambda row: row.str.capitalize()
    )
    results_df['vqa_strategy_type'] = results_df.apply(
        lambda row: _get_pretty_strategy_representation(
            row['vqa_strategy_type'], row['should_apply_rag_to_question']
        ),
        axis=1
    )
    results_df['add_title'] = results_df['add_title'].apply(_transform_add_title)
    results_df.drop(columns=['doc_splitter', 'well_formatted_answers'], inplace=True)
    filtered_results_df = results_df.drop(index=excluded_indexes)
    max_accuracy_value = filtered_results_df['accuracy'].max()
    max_accuracy_rows = results_df[results_df['accuracy'] == max_accuracy_value].index
    column_mapping = {
        'country': 'Country',
        'file_type': 'File Type',
        'vqa_strategy_type': 'VQA Strategy',
        'prompt_type': 'Prompt Type',
        'relevant_docs_count': 'Relevant Document Count',
        'add_title': 'Title',
        'token_count': 'Token Count',
        'chunk_size': 'Chunk Size',
        'accuracy': 'Accuracy'
    }
    results_df = results_df.rename(columns=column_mapping)[column_mapping.values()]

    header_style = {
        'selector': 'thead th', 
        'props': [
            ('background-color', '#0056A6'),
            ('color', '#FFFFFF'),
            ('font-weight', 'bold'),
            ('padding', '12px'),
            ('text-align', 'center')
        ]
    }
    odd_row_style = {
        'selector': 'tbody tr:nth-child(odd)',
        'props': [
            ('background-color', '#e0e0e0'),
            ('color', '#000000')
        ]
    }
    even_row_style = {
        'selector': 'tbody tr:nth-child(even)',
        'props': [
            ('background-color', '#FFFFFF'),
            ('color', '#000000')
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
        'props': [('border', '1px solid #BBBBBB')]
    }
    highlight_styles = [
        {
            "selector": f"tbody tr:nth-child({i + 1})",
            "props": [
                ('background-color', '#FFD700'),
                ('color', '#000000'),
            ]
        }
        for i in max_accuracy_rows
    ]
    world_med_qa_v_highlight_style = {
        "selector": f"tbody tr:nth-child({1})",
        "props": [
            ('background-color', '#006400'),
            ('color', '#FFFFFF')
        ]
    }


    table_styles = [
        header_style,
        odd_row_style,
        even_row_style,
        padding_and_text_alignment,
        table_style,
        border_style,
    ] + highlight_styles
    if include_green_highlight:
        table_styles.append(world_med_qa_v_highlight_style)

    styled_results_df = results_df.style.set_table_styles(table_styles).format({
        'Accuracy': '{:.4f}',
    }).hide(axis='index')

    display(styled_results_df)


def display_test_results_summary(
    evaluation_results_list: list[pd.DataFrame]
) -> None:
    test_results_df = pd.concat(evaluation_results_list, ignore_index=True)

    # Fill the results dataframe
    test_results_summary_df = pd.DataFrame({
        '': test_results_df['country'].unique(),
        'Zero-Shot': test_results_df[
            test_results_df['vqa_strategy_type'] == VQAStrategyType.ZERO_SHOT.value
        ]['accuracy'].tolist(),
        'RAG (Question Only)': test_results_df[
            test_results_df['vqa_strategy_type'] == VQAStrategyType.RAG_Q.value
        ]['accuracy'].tolist(),
        'RAG (Answers Only)': test_results_df[
            (test_results_df['vqa_strategy_type'] == VQAStrategyType.RAG_Q_AS.value) &
            (test_results_df['should_apply_rag_to_question'].eq(False))
        ]['accuracy'].tolist(),
        'RAG (Question and Answers)': test_results_df[
            (test_results_df['vqa_strategy_type'] == VQAStrategyType.RAG_Q_AS.value) &
            (test_results_df['should_apply_rag_to_question'].eq(True))
        ]['accuracy'].tolist()
    })
    test_results_summary_df.loc[len(test_results_summary_df)] = {
        '': 'Mean Accuracy',
        'Zero-Shot': test_results_summary_df['Zero-Shot'].mean(),
        'RAG (Question Only)': test_results_summary_df['RAG (Question Only)'].mean(),
        'RAG (Answers Only)': test_results_summary_df['RAG (Answers Only)'].mean(),
        'RAG (Question and Answers)': test_results_summary_df['RAG (Question and Answers)'].mean()
    }

    # Format the content of the dataframe
    test_results_summary_df[''] = test_results_summary_df[''].str.title()

    best_accuracy_scores, second_best_accuracy_scores = [], []
    numeric_test_results_df = test_results_summary_df.iloc[:, 1:]
    for index, row in numeric_test_results_df.iterrows():
        highest_indices, second_highest_indices = _get_top2_accuracy_column_indices(row)
        best_accuracy_scores += [
            (index, highest_index)
            for highest_index in highest_indices
        ]
        second_best_accuracy_scores += [
            (index, second_highest_index)
            for second_highest_index in second_highest_indices
        ]

    header_style = {
        'selector': 'thead th', 
        'props': [
            ('background-color', '#0056A6'),
            ('color', '#FFFFFF'),
            ('font-weight', 'bold'),
            ('padding', '12px'),
            ('text-align', 'center')
        ]
    }
    odd_row_style = {
        'selector': 'tbody tr:nth-child(odd)',
        'props': [
            ('background-color', '#e0e0e0'),
            ('color', '#000000')
        ]
    }
    even_row_style = {
        'selector': 'tbody tr:nth-child(even)',
        'props': [
            ('background-color', '#FFFFFF'),
            ('color', '#000000')
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
        'props': [('border', '1px solid #BBBBBB')]
    }
    separator_style = {
        "selector": f"tbody tr:nth-child({len(test_results_summary_df) - 1})",
        "props": [("border-bottom", "3px solid #444444")]
    }
    highlight_best_styles = [
        {
            "selector": f"tbody tr:nth-child({row_idx + 1}) td:nth-child({col_idx + 2})",
            "props": [
                ('background-color', '#4CAF50'),
                ('color', '#FFFFFF'),
                ('font-weight', 'bold')
            ]
        }
        for (row_idx, col_idx) in best_accuracy_scores
    ]
    highlight_second_best_styles = [
        {
            "selector": f"tbody tr:nth-child({row_idx + 1}) td:nth-child({col_idx + 2})",
            "props": [
                ('background-color', '#A5D6A7'),
                ('font-weight', 'bold')
            ]
        }
        for (row_idx, col_idx) in second_best_accuracy_scores
    ]
    table_styles = [
        header_style,
        odd_row_style,
        even_row_style,
        padding_and_text_alignment,
        table_style,
        border_style,
        separator_style,
        *highlight_best_styles,
        *highlight_second_best_styles
    ]
    styled_test_results_summary_df = (
        test_results_summary_df.style.set_table_styles(table_styles).format({
            'Zero-Shot': '{:.4f}',
            'RAG (Question Only)': '{:.4f}',
            'RAG (Answers Only)': '{:.4f}',
            'RAG (Question and Answers)': '{:.4f}'
        }).hide(axis='index')
    )

    display(styled_test_results_summary_df)


# ====================
# Private Functions
# ====================


def _get_subsets_data_by_split_type(
    splits_data: dict[str, Dataset],
    split_type: str
) -> dict[str, Any]:
    return dict(filter(lambda subset: subset[0].endswith(split_type), splits_data.items()))


def _get_formatted_subset_prefix(subset_name: str) -> str:
    return subset_name.split('_')[0].capitalize()


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


def _get_splitter_results_column_names(
    splitter_evaluation_results: pd.DataFrame,
    document_splitter_type: DocumentSplitterType
) -> list:
    column_names = []

    for _, row in splitter_evaluation_results.iterrows():
        column_name_elements = []

        if row['add_title']:
            column_name_elements.append("with_title")
        else:
            column_name_elements.append("no_title")

        column_name_elements.append(f"tc{row['token_count']}")

        if document_splitter_type == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER:
            column_name_elements.append(f"cs{row['chunk_size']}")

        column_names.append("_".join(column_name_elements))

    return column_names


def _get_top2_accuracy_column_indices(row: pd.Series) -> tuple[list[int], list[int]]:
    unique_top_values = row.sort_values(ascending=False).unique()
    highest_value = unique_top_values[0]
    second_highest_value = unique_top_values[1] if len(unique_top_values) > 1 else None

    highest_indices = [row.index.get_loc(col) for col, val in row.items() if val == highest_value]
    if second_highest_value is not None:
        second_highest_indices = [
            row.index.get_loc(col)
            for col, val in row.items() if val == second_highest_value
        ]
    else:
        second_highest_indices = []

    return highest_indices, second_highest_indices
