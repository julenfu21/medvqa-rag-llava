import base64
from io import BytesIO
from typing import Optional

from IPython.display import display, HTML
from PIL import Image


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
