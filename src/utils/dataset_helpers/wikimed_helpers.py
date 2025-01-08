import json
from pathlib import Path

import pandas as pd
import plotly.express as px
from IPython.display import display
from tqdm import tqdm

from src.utils.enums import WikiMedRepresentationMode


def _calculate_summary_statistics(column: pd.Series) -> dict[str, float]:
    return {
        "Min": column.min(),
        "Q1": column.quantile(0.25),
        "Median": column.quantile(0.5),
        "Q3": column.quantile(0.75),
        "Max": column.max()
    }


def _classify_quartile_interval(value: float, quartiles: dict[str, float]) -> str:
    iqr = quartiles['Q3'] - quartiles['Q1']

    if value < quartiles['Q1']:
        return "[Min., Q1)"
    if value < quartiles['Median']:
        return "[Q1, Q2)"
    if value < quartiles['Q3']:
        return "[Q2, Q3)"
    if value < quartiles['Q3'] + 1.5 * iqr:
        return "[Q3, Max.)"
    return "Outlier"


def _add_quartile_intervals(data_frame: pd.DataFrame, column_names: list[str]) -> None:
    for column_name in column_names:
        summary_stats = _calculate_summary_statistics(
            column=data_frame[column_name]
        )
        new_column_name = f"{column_name.split('_')[0]}_quartile_interval"
        data_frame[new_column_name] = data_frame[column_name].apply(
            lambda x: _classify_quartile_interval(x, summary_stats)
        )
    return data_frame


def load_wikimed_dataset_metadata(data_path: Path) -> pd.DataFrame:
    wikimed_dataset_metadata = []

    with open(file=data_path, mode="r", encoding="utf-8") as wikimed_file:
        wikimed_file.seek(0, 2)
        wikimed_length = wikimed_file.tell()
        wikimed_file.seek(0)

        with tqdm(
            total=wikimed_length,
            desc="- Loading WikiMed dataset metadata ...",
            unit="B",
            unit_scale=True
        ) as progress_bar:
            for line in wikimed_file:
                line_content = json.loads(line)
                line_id = line_content['_id']
                line_text = line_content['text']

                wikimed_dataset_metadata.append({
                    "id": int(line_id),
                    "word_count": len(line_text),
                    "sentence_count": len(line_text.split('.'))
                })
                progress_bar.update(len(line.encode('utf-8')))

    wikimed_dataset_metadata_df = _add_quartile_intervals(
        data_frame=pd.DataFrame(wikimed_dataset_metadata),
        column_names=['word_count', 'sentence_count']
    )
    print("+ WikiMed dataset metadata loaded.")
    return wikimed_dataset_metadata_df


def display_boxplot(data_frame: pd.DataFrame, title: str, x_column: str) -> None:
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


def display_bar_chart_on_wikimed_data(
    data_frame: pd.DataFrame,
    title: str,
    wikimed_representation_mode: WikiMedRepresentationMode,
    categories: list[str]
) -> None:
    representation_mode_count_column = f"{wikimed_representation_mode.value}_count"
    summary_stats = _calculate_summary_statistics(
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
