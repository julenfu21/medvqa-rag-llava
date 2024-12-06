import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Optional

from IPython.display import HTML, display
from PIL import Image
from datasets import Dataset


def resize_image(
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


def display_base64_image(
    base64_image: str,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> None:
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    resized_image = resize_image(image, width, height)
    display(resized_image)


def display_formatted_section(
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


def get_dataset_row_by_id(
    dataset: Dataset,
    question_id: int
) -> dict:
    filtered_dataset = dataset.filter(lambda row: row['index'] == question_id)
    if len(filtered_dataset) == 0:
        raise ValueError(f"No row found with index {question_id}")
    return filtered_dataset[0]


def visualize_qa_pair_row(
    row: dict,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    model_answer: str = None,
) -> None:
    # Display row id
    display_formatted_section(
        section_name="ID",
        section_style="margin: 20px 0;",
        section_content=row['index']
    )

    # Display question
    display_formatted_section(
        section_name="Question",
        section_style="margin-bottom: 20px;",
        section_content=row['question']
    )

    # Display context image
    display_formatted_section(
        section_name="Context Image",
        section_style="margin-bottom: 20px;",
        section_content=""
    )
    display_base64_image(
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
        elif option == model_answer:
            formatted_options.append(
                f"<p style='color: rgb(255, 0, 0);'><b>{option}) {row[option]}</b>"
            )
        else:
            formatted_options.append(f"<p>{option}) {row[option]}")
    answer = "<br><br>" + "<br>".join(formatted_options)

    display_formatted_section(
        section_name="Possible Answers",
        section_style="margin-top: 30px;",
        section_content=answer
    )

    if model_answer:
        display_formatted_section(
            section_name="Model Answer",
            section_style="margin: 30px 0;",
            section_content=model_answer
        )


def fetch_model_answer_from_json(
    evaluation_results_folder: Path,
    question_id: int,
    vqa_strategy_name: str
) -> str:
    evaluation_results_filename = f'spain_english_{vqa_strategy_name}_evaluation.json'
    evaluation_results_path = evaluation_results_folder / evaluation_results_filename
    with open(evaluation_results_path, mode='r', encoding='utf-8') as evaluation_file:
        evaluation_data = json.load(evaluation_file)

    return evaluation_data['predictions'][str(question_id)]['predicted_answer']
