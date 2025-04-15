import ipywidgets as widgets

from src.ui.utils.data_definitions import WidgetWrapper


def create_dropdown(
    description: str,
    options: dict,
    disabled: bool = False,
) -> WidgetWrapper:
    initial_value = list(options.values())[0]

    return WidgetWrapper(
        widget=widgets.Dropdown(
            description=description,
            options=options,
            value=initial_value,
            layout=widgets.Layout(width="100%"),
            style={"description_width": "33%"},
            disabled=disabled
        ),
        initial_value=initial_value
    )

def create_int_widget(
    description: str,
    disabled: bool = False,
    initial_value: int = 0,
    min_value: int = 0,
    max_value: int = 100,
    step: int = 1
) -> WidgetWrapper:
    return WidgetWrapper(
        widget=widgets.BoundedIntText(
            description=description,
            value=initial_value,
            min=min_value,
            max=max_value,
            step=step,
            layout=widgets.Layout(width="100%"),
            style={"description_width": "33%"},
            disabled=disabled
        ),
        initial_value=initial_value
    )

def create_checkbox(
    description: str,
    initial_value: bool,
    disabled: bool = False
) -> WidgetWrapper:
    return WidgetWrapper(
        widget=widgets.Checkbox(
            description=description,
            value=initial_value,
            indent=False,
            layout=widgets.Layout(width="auto", margin="8px auto"),
            disabled=disabled
        ),
        initial_value=initial_value
    )

def create_button(
    description: str,
    icon: str,
    tooltip: str,
    button_style: str,
    width: str = "120px",
    height: str = "36px"
) -> widgets.Button:
    return widgets.Button(
        description=description,
        icon=icon,
        tooltip=tooltip,
        button_style=button_style,
        layout=widgets.Layout(
            width=width,
            height=height,
            margin="0 auto"
        ),
        style={
            "font_weight": "bold",
            "font_size": "14px"
        }
    )

def create_text_field(
    description: str,
    placeholder: str,
    disabled: bool = False,
    initial_value: str = ""
) -> WidgetWrapper:
    return WidgetWrapper(
        widget=widgets.Text(
            description=description,
            value=initial_value,
            placeholder=placeholder,
            layout=widgets.Layout(width="100%"),
            style={"description_width": "33%"},
            disabled=disabled
        ),
        initial_value=initial_value
    )
