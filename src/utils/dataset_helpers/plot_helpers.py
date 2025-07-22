from collections import defaultdict
from itertools import product

import pandas as pd
import plotly.express as px
from datasets import Dataset
from IPython.display import display
from langchain_ollama import ChatOllama


def display_average_document_length_in_context(
    wikimed_dataset_metadata: pd.DataFrame,
    world_med_qa_v_dataset: dict[str, Dataset]
) -> None:
    # Calculate the average question length for each split within the WorldMedQA-V dataset
    llava_model = ChatOllama(model="llava", temperature=0, num_predict=1)
    average_question_lengths = defaultdict(float)
    for split_name, split_data in world_med_qa_v_dataset.items():
        for question in split_data['question']:
            average_question_lengths[split_name] += llava_model.get_num_tokens(question)
        average_question_lengths[split_name] /= len(split_data['question'])

    # Prepare data to be displayed in the grouped stacked bar-graph
    countries = ['brazil', 'israel', 'japan', 'spain']
    subset_types = ['local', 'english']
    average_token_count_per_document = wikimed_dataset_metadata['model_token_count'].mean()
    data = {
        'x': ['Original', 'English'] * 8,
        'y': [
            average_question_lengths[f'{country}_{subset_type}']
            for country, subset_type in product(countries, subset_types)
        ] + [*[average_token_count_per_document] * 8],
        'color': [*["Original Language", "English Translation"] * 4, *["Extra Tokens"] * 8],
        'facet_col': [
            country.capitalize()
            for country in countries
            for _ in range(2)
        ] * 2
    }

    # Create grouped stacked bar-graph
    antique_colors = px.colors.qualitative.Antique
    custom_colors = {
        "Original Language": antique_colors[0],
        "English Translation": antique_colors[1],
        "Extra Tokens": antique_colors[2]
    }
    grouped_stacked_bar_graph = px.bar(
        data,
        x="x",
        y="y",
        color="color",
        facet_col="facet_col",
        color_discrete_map=custom_colors,
        custom_data=["facet_col"]
    )

    # Modify x and y-axis appearance
    domains = [[0.03, 0.22], [0.28, 0.47], [0.53, 0.72], [0.78, 0.97]]
    y_min, y_max = 0, 2300
    for (index, domain), country in zip(enumerate(domains), countries):
        grouped_stacked_bar_graph.update_xaxes(
            title_text=country.capitalize(),
            domain=domain,
            row=1,
            col=index + 1,
            showticklabels=False
        )
    grouped_stacked_bar_graph.update_yaxes(
        anchor='free',
        position=0,
        range=[y_min, y_max],
        title_text="Average Context Length (in Tokens)",
        title_font={
            'size': 16,
            'color': 'black'
        },
        ticks="outside",
        showgrid=False,
        row=1,
        col=1
    )
    for column_index in range (2, 5):
        grouped_stacked_bar_graph.update_yaxes(showgrid=False, row=1, col=column_index)

    # Modify general appearance
    grouped_stacked_bar_graph.for_each_annotation(lambda a: a.update(text=''))
    grouped_stacked_bar_graph.update_traces(
        width=1,
        hovertemplate=(
            "<b>Subset</b>: %{customdata[0]}<br>"
            "<b>Avg. Length</b>: %{y}"
        )
    )
    grouped_stacked_bar_graph.update_layout(
        legend={
            'title': 'Subset Type',
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.35,
            'xanchor': 'center',
            'x': 0.5,
            'font': {'size': 14}
        },
        width=850,
        height=650,
        margin={'t': 100},
        title={
            'x': 0.5,
            'font': {
                'size': 24,
                'color': 'black'
            },
            'text': "Average Document Length in Context"
        },
    )

    ## Other Modifications

    # Wrap bar graph with a black box
    grouped_stacked_bar_graph.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        line={
            'color': 'black',
            'width': 2
        },
        layer="above"
    )

    # Modify background of the bar graph
    grouped_stacked_bar_graph.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        fillcolor="#E5ECF6",
        layer="below",
        line_width=0
    )

    # Add white horizontal lines every tick in the y-axis
    tick_vals = list(range(y_min, y_max, 500))
    for y in tick_vals:
        grouped_stacked_bar_graph.add_shape(
            type="line",
            xref="paper", yref="y",
            x0=0,
            x1=1,
            y0=y,
            y1=y,
            line={
                'color': 'white',
                'width': 1
            },
            layer="below"
        )

    # Add 'Splits' annotation to the bottom of the x-axis
    grouped_stacked_bar_graph.add_annotation(
        text="Splits",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.18,
        showarrow=False,
        font={
            'size': 16,
            'color': 'black'
        },
        align="center"
    )

    # Add red dotted line
    grouped_stacked_bar_graph.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=2048,
        y1=2048,
        line={
            'color': 'red',
            'width': 2,
            'dash': 'dash'
        },
        layer="above"
    )

    # Add annotation above the red dotted line
    grouped_stacked_bar_graph.add_annotation(
        text="<b>Default Context Window: 2048 Tokens</b>",
        xref="paper",
        yref="y",
        x=0.61,
        y=2048,
        showarrow=False,
        font={
            'size': 14,
            'color': 'red'
        },
        align="right",
        xanchor="left",
        yanchor="bottom"
    )

    display(grouped_stacked_bar_graph)
