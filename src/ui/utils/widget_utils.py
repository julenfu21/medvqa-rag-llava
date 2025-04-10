from typing import Any, Union

import ipywidgets as widgets

from src.ui.utils.data_definitions import DependentWidgetsConfig


def update_dependent_dropdown_values(
    change: dict[str, Any],
    dependent_dropdown: widgets.Dropdown,
    possible_options: dict[Any, list[Any]]
) -> None:
    main_dropdown_selected_value = change['new']
    dependent_dropdown.options = possible_options[main_dropdown_selected_value]
    dependent_dropdown.value = list(dependent_dropdown.options.values())[0]

def update_dependent_int_widget_values(
    change: dict[str, Any],
    dependent_int_widget: widgets.BoundedIntText,
    possible_values: dict[Any, int]
) -> None:
    main_dropdown_selected_value = change['new']
    dependent_int_widget.max = possible_values[main_dropdown_selected_value]

def update_dependent_widgets_state(
    dependent_widgets_config: list[DependentWidgetsConfig]
) -> None:
    for config in dependent_widgets_config:
        for widget in config.widgets:
            widget.disabled = not config.enable_condition


def get_widget_value(widget: widgets.Widget) -> Union[int, str]:
    return None if widget.disabled else widget.value
