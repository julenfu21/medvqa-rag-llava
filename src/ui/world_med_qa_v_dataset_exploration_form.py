import ipywidgets as widgets
from datasets import Dataset

import src.utils.dataset_helpers.world_med_qa_v.dataset_management as world_med_qa_v_dataset_management
from src.ui.base_interactive_form import BaseInteractiveForm
from src.ui.utils.display_utils import visualize_qa_pair_row
from src.ui.utils.widget_utils import update_dependent_int_widget_values
from src.ui.widgets.widget_factory import (
    create_checkbox,
    create_dropdown,
    create_int_widget
)


class WorldMedQAVDatasetExplorationForm(BaseInteractiveForm):

    def __init__(self, dataset: dict[str, Dataset]) -> None:
        # Options Layout elements
        self.__general_options_layout = None
        self.__country_dropdown = None
        self.__file_type_dropdown = None
        self.__question_id_int_widget = None
        self.__use_image_checkbox = None

        # Create Form
        super().__init__(form_title="WorldMedQA-V Dataset Exploration Form", dataset=dataset)


    def _get_initial_output_widget_text(self) -> str:
        return """
        This interactive form lets you explore the <b>WorldMedQA-V Dataset</b> by adjusting various options. You can:

        <span style='margin-left: 30px;'>- Set <b>key inputs</b>, such as country, file type and question ID.</span>

        <span style='margin-left: 30px;'>- <b>Toggle context image display</b> to view any associated visual information that might be provided to a model.</span>

        Selected options are displayed in an organized format, making it easy to explore different dataset subsets.

        <i>Note: You can revisit this information anytime by clicking the reset button.</i>
        """


    def _create_options_accordion(self) -> widgets.Accordion:
        self.__general_options_layout = self.__create_general_options_layout()

        return widgets.Accordion(
            children=[
                self.__general_options_layout
            ],
            titles=[
                "General Options"
            ],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

    def __create_general_options_layout(self) -> widgets.VBox:
        self.__country_dropdown = create_dropdown(
            description='Country:',
            options={
                'Spain 🇪🇸': 'spain',
                'Brazil 🇧🇷': 'brazil',
                'Israel 🇮🇱': 'israel',
                'Japan 🇯🇵': 'japan'
            }
        )
        self.__file_type_dropdown = create_dropdown(
            description='File Type:',
            options={
                'English Translation': 'english',
                'Original Language': 'local'
            }
        )
        self.__question_id_int_widget = create_int_widget(
            description="Question ID:",
            initial_value=1,
            min_value=1,
            max_value=self.__get_max_question_id_for_dataset_split(),
            step=1
        )
        self.__use_image_checkbox = create_checkbox(
            description="Use Image",
            initial_value=True
        )

        return widgets.VBox(
            children=[
                self.__country_dropdown.widget,
                self.__file_type_dropdown.widget,
                self.__question_id_int_widget.widget,
                widgets.HBox(
                    children=[
                        self.__use_image_checkbox.widget
                    ],
                    layout=widgets.Layout(
                        width="100%",
                        overflow="visible",
                        display="flex",
                        justify_content="center"
                    )
                )
            ],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

    def __get_max_question_id_for_dataset_split(self) -> int:
        dataset_name = (
            f"{self.__country_dropdown.widget.value}_"
            f"{self.__file_type_dropdown.widget.value}"
        )
        return max(self._dataset[dataset_name]['index'])


    def _get_initial_options_output_widget_text(self) -> str:
        return ""


    def _add_callbacks(self) -> None:
        self.__country_dropdown.widget.observe(
            lambda _: update_dependent_int_widget_values(
                dependent_int_widget=self.__question_id_int_widget.widget,
                max_value=self.__get_max_question_id_for_dataset_split()
            ),
            names='value'
        )


    def _run_form(self) -> None:
        self._output_widget_manager.clear_content()

        question_id = self.__question_id_int_widget.widget.value
        dataset_name = (
            f"{self.__country_dropdown.widget.value}_"
            f"{self.__file_type_dropdown.widget.value}"
        )

        try:
            row = world_med_qa_v_dataset_management.get_dataset_row_by_id(
                dataset=self._dataset[dataset_name],
                question_id=question_id
            )
        except ValueError:
            self._output_widget_manager.display_text_content(
                content=f"""
                ⚠️ Unable to find a question with ID: {question_id} ⚠️
                
                Make sure the ID exists in the {dataset_name} dataset subset.
                """,
                extra_css_style="color: #b71c1c; font-weight: bold;"
            )
        else:
            visualize_qa_pair_row(
                output_widget_manager=self._output_widget_manager,
                display_image=self.__use_image_checkbox.widget.value,
                row=row,
                model_answer_result=None
            )


    def _reset_form_specific_widgets(self) -> None:
        super()._reset_widgets(
            widget_wrappers=[
                self.__country_dropdown,
                self.__file_type_dropdown,
                self.__question_id_int_widget,
                self.__use_image_checkbox
            ]
        )
