from abc import ABC, abstractmethod

import ipywidgets as widgets
from datasets import Dataset
from IPython.display import display

from src.ui.utils.data_definitions import WidgetWrapper
from src.ui.widgets.output_widget_manager import OutputWidgetManager
from src.ui.widgets.widget_factory import create_button



class BaseInteractiveForm(ABC):

    def __init__(self, form_title: str, dataset: dict[str, Dataset]) -> None:
        self.__form_title = form_title
        self._dataset = dataset

        # Main Layout Elements
        self.__root_widget = None
        self.__title_widget = None
        self._output_widget_manager = None
        self.__options_layout = None

        # Layout elements to interact with the form state
        self.__options_accordion = None
        self._options_output_widget_manager = None
        self.__buttons_layout = None

        # Form buttons
        self.__run_button = None
        self.__reset_button = None

        # Create Layout and add Functionality
        self.__create_layout()
        self._add_callbacks()


    def display_form(self) -> None:
        display(self.__root_widget)


    # ============================
    # Layout Creation Methods
    # ============================

    def __create_layout(self) -> None:
        self.__title_widget = widgets.HTML(
            value=(
                "<h1 style='text-align: center; margin-bottom: 20px;'>"
                f"{self.__form_title}"
                "</h1>"
            )
        )
        self._output_widget_manager = OutputWidgetManager(
            initial_content=self._get_initial_output_widget_text(),
            width="50%"
        )
        self.__options_layout = self.__create_options_layout()

        self.__root_widget = widgets.VBox(
            children=[
                self.__title_widget,
                widgets.HBox(
                    children=[
                        self._output_widget_manager.output_widget,
                        self.__options_layout
                    ],
                    layout=widgets.Layout(
                        width="100%",
                        align_items="stretch",
                        overflow="visible",
                        padding="20px"
                    )
                )
            ]
        )

    @abstractmethod
    def _get_initial_output_widget_text(self) -> str:
        pass

    def __create_options_layout(self) -> widgets.VBox:
        self.__options_accordion = self._create_options_accordion()
        self.__buttons_layout = self.__create_buttons_layout()
        self._options_output_widget_manager = OutputWidgetManager(
            initial_content=self._get_initial_options_output_widget_text(),
            width="100%"
        )

        return widgets.VBox(
            children=[
                self.__options_accordion,
                self.__buttons_layout,
                self._options_output_widget_manager.output_widget
            ],
            layout=widgets.Layout(
                width="50%",
                overflow="visible",
                display="flex",
                flex_flow="column",
                align_items="stretch"
            )
        )

    @abstractmethod
    def _create_options_accordion(self) -> widgets.Accordion:
        pass

    def __create_buttons_layout(self) -> widgets.HBox:
        self.__run_button = create_button(
            description="Run",
            icon="play",
            tooltip="Run the form with the selected options",
            button_style="success"
        )
        self.__run_button.on_click(lambda _: self._run_form())

        self.__reset_button = create_button(
            description="Reset",
            icon="undo",
            tooltip="Reset all form inputs to default",
            button_style="warning"
        )
        self.__reset_button.on_click(lambda _: self.__reset_form())

        return widgets.HBox(
            children=[
                self.__run_button,
                self.__reset_button
            ],
            layout=widgets.Layout(
                width="100%",
                overflow="visible",
                display="flex",
                justify_content="center",
                margin="30px 0"
            )
        )

    @abstractmethod
    def _get_initial_options_output_widget_text(self) -> str:
        pass


    # =============================
    # Form Interaction Methods
    # =============================

    @abstractmethod
    def _add_callbacks(self) -> None:
        pass


    @abstractmethod
    def _run_form(self) -> None:
        pass


    def __reset_form(self) -> None:
        self._output_widget_manager.reset_content()
        self._options_output_widget_manager.reset_content()

        self._reset_form_specific_widgets()

    @abstractmethod
    def _reset_form_specific_widgets(self) -> None:
        pass

    @staticmethod
    def _reset_widgets(widget_wrappers: list[WidgetWrapper]) -> None:
        for widget_wrapper in widget_wrappers:
            widget_wrapper.reset_widget()
