from dataclasses import dataclass
from typing import Any, Union

import ipywidgets as widgets


@dataclass
class DependentWidgetsConfig:
    widgets: list[widgets.Widget]
    enable_condition: bool


@dataclass
class WidgetWrapper:
    widget: Union[widgets.Dropdown, widgets.BoundedIntText, widgets.Checkbox]
    initial_value: Any

    def reset_widget(self) -> None:
        self.widget.value = self.initial_value
