import pandas as pd
import plotly.express as px
from IPython.display import display

from src.utils.enums import WikiMedRepresentationMode
from src.utils.dataset_helpers.wikimed.dataset_management import calculate_summary_statistics


def display_boxplot_on_column(data_frame: pd.DataFrame, title: str, x_column: str) -> None:
    boxplot = px.box(data_frame, x=x_column)

    boxplot.update_traces(
        boxmean=True,
        boxpoints='suspectedoutliers',
        marker={
            'color': 'rgba(255, 99, 71, 1)'
        },
        line={
            'color': 'rgba(72, 61, 139, 0.5)',
            'width': 2
        }
    )

    xaxis_title = " ".join([word.capitalize() for word in x_column.split('_')])
    boxplot.update_layout(
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
        xaxis_title=xaxis_title,
        margin={
            'l': 50,
            'r': 50,
            't': 100,
            'b': 100
        }
    )

    display(boxplot)


def display_bar_chart_on_documents_length(
    data_frame: pd.DataFrame,
    title: str,
    wikimed_representation_mode: WikiMedRepresentationMode,
    categories: list[str]
) -> None:
    representation_mode_count_column = f"{wikimed_representation_mode.value}_count"
    summary_stats = calculate_summary_statistics(
        column=data_frame[representation_mode_count_column]
    )
    bar_chart_x_column = f"{wikimed_representation_mode.value}_quartile_interval"

    bar_chart = px.bar(
        data_frame.groupby(bar_chart_x_column).size().reset_index(name='category_count'),
        x=bar_chart_x_column,
        y="category_count",
        title=title,
        category_orders={bar_chart_x_column: categories},
        color=bar_chart_x_column,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    xaxis_title = " ".join([word.capitalize() for word in bar_chart_x_column.split('_')])
    bar_chart.update_layout(
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
        xaxis_title=xaxis_title,
        yaxis_title="Frequency",
        font={
            'family': "Arial, sans-serif",
            'size': 14,
            'color': "black"
        },
        barcornerradius=15,
        legend_title=xaxis_title,
        height=500,
        annotations=[
            {
                'x': 1.020,
                'y': 0.45,
                'xref': 'paper',
                'yref': 'paper',
                'text': "Quartile values:",
                'font': {
                    'size': 17,
                    'color': 'black',
                    'family': 'Arial, sans-serif'
                },
                'showarrow': False,
                'xanchor': 'left',
            },
            {
                'x': 1.020,
                'y': 0.30,
                'xref': 'paper',
                'yref': 'paper',
                'text': f"  Q1: {summary_stats['Q1']:.2f}",
                'showarrow': False,
                'xanchor': 'left',
            },
            {
                'x': 1.020,
                'y': 0.24,
                'xref': 'paper',
                'yref': 'paper',
                'text': f"  Q2 (Median): {summary_stats['Median']:.2f}",
                'showarrow': False,
                'xanchor': 'left',
            },
            {
                'x': 1.020,
                'y': 0.18,
                'xref': 'paper',
                'yref': 'paper',
                'text': f"  Q3: {summary_stats['Q3']:.2f}",
                'showarrow': False,
                'xanchor': 'left',
            },
            {
                'x': 1.020,
                'y': 0.06,
                'xref': 'paper',
                'yref': 'paper',
                'text': f"  Min: {summary_stats['Min']:.2f}",
                'showarrow': False,
                'xanchor': 'left',
            },
            {
                'x': 1.020,
                'y': 0.00,
                'xref': 'paper',
                'yref': 'paper',
                'text': f"  Max: {summary_stats['Max']:.2f}",
                'showarrow': False,
                'xanchor': 'left',
            }
        ]
    )

    display(bar_chart)
