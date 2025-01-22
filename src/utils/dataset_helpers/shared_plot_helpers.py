from IPython.display import HTML, display


def _display_formatted_section(
    section_name: str,
    section_style: str,
    section_content: str
) -> None:
    section_text = f"""
    <div style='{section_style}'>
        <b>{section_name}:</b> {section_content.replace('\n', '<br>')}
    </div>
    """
    display(HTML(section_text))
