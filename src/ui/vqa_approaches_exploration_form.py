from pathlib import Path
from typing import Any, Optional, Union

import ipywidgets as widgets
from datasets import Dataset
from IPython.display import display

import src.utils.dataset_helpers.world_med_qa_v.dataset_management as world_med_qa_v_dataset_management
from src.utils.enums import (
    DocumentSplitterType,
    RagQPromptType,
    VQAStrategyType,
    ZeroShotPromptType
)
from src.ui.output_widget_manager import OutputWidgetManager
from src.utils.data_definitions import (
    DependentWidgetsConfig,
    DocSplitterOptions,
    ModelAnswerResult,
    VQAStrategyDetail
)
from src.utils.string_formatting_helpers import (
    prettify_document_splitter_name, prettify_strategy_name
)
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy
from src.visual_qa_model import VisualQAModel


class VQAAproachesExplorationForm:

    def __init__(
        self,
        dataset: dict[str, Dataset],
        model_name: str,
        vqa_strategies: dict[VQAStrategyType, BaseVQAStrategy],
        evaluation_results_folder: Path
    ) -> None:
        # Dataset and Model Configurations
        self.__world_med_qa_v_dataset = dataset
        self.__model_name = model_name
        self.__vqa_strategies = vqa_strategies
        self.__evaluation_results_folder = evaluation_results_folder

        # Main Layout Elements
        self.__title_widget = None
        self.__options_layout = None
        self.__options_accordion = None
        self.__output_widget_manager = None
        self.__options_output_widget_manager = None

        # General Options Widgets
        self.__general_options_layout = None
        self.__action_type_dropdown = None
        self.__country_dropdown = None
        self.__file_type_dropdown = None
        self.__question_id_int_widget = None
        self.__vqa_strategy_type_dropdown = None
        self.__prompt_type_dropdown = None
        self.__use_image_checkbox = None

        # RAG Options Widgets
        self.__rag_options_layout = None
        self.__relevant_documents_count_int_widget = None
        self.__apply_rag_to_question_checkbox = None

        # Document Splitter Options Widgets
        self.__document_splitter_options_layout = None
        self.__document_splitter_type_dropdown = None
        self.__token_count_int_widget = None
        self.__chunk_size_int_widget = None
        self.__chunk_overlap_int_widget = None
        self.__add_title_checkbox = None

        # Form Buttons
        self.__buttons_layout = None
        self.__run_button = None
        self.__reset_button = None

        # Create UI
        self.__root_widget = None
        self.__create_layout()
        self.__add_callbacks()

    def visualize(self) -> None:
        display(self.__root_widget)


    # ====================
    # Private Functions
    # ====================


    def __create_layout(self) -> None:
        self.__create_general_options_layout()
        self.__create_rag_options_layout()
        self.__create_buttons_layout()

        self.__options_accordion = widgets.Accordion(
            children=[
                self.__general_options_layout,
                self.__rag_options_layout
            ],
            titles=[
                "General Options",
                "RAG Options"
            ],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

        self.__options_output_widget_manager = OutputWidgetManager(
            initial_content="""
            Here the options that have been selected will appear.

            These options include general settings, RAG-specific parameters, and other relevant selections. An example of the default selected options is shown below:

            <b>Specified Options:</b>
                <b style='margin-left: 30px;'>+ General Options:</b>
                    <span style='margin-left: 60px;'>- Country: Spain 🇪🇸</span>
                    <span style='margin-left: 60px;'>- File Type: English Translation</span>
                    <span style='margin-left: 60px;'>- Question ID: 1</span>
                    <span style='margin-left: 60px;'>- VQA Strategy Type: Zero-Shot</span>
                    <span style='margin-left: 60px;'>- Prompt Type: zs_v1</span>
                    <span style='margin-left: 60px;'>- Use Image: ✅</span>
            """,
            width="100%"
        )
        self.__options_layout = widgets.VBox(
            children=[
                self.__options_accordion,
                self.__buttons_layout,
                self.__options_output_widget_manager.output_widget
            ],
            layout=widgets.Layout(
                width="50%",
                overflow="visible",
                display="flex",
                flex_flow="column",
                align_items="stretch"
            )
        )

        self.__output_widget_manager = OutputWidgetManager(
            initial_content="""
            This interactive form lets you configure and explore different <b>Visual Question Answering (VQA) strategies</b> by adjusting various options. You can:

            <span style='margin-left: 30px;'>- Set <b>key inputs</b>, like country, file type, question ID, and VQA strategy.</span>

            <span style='margin-left: 30px;'>- <b>Customize document processing</b>, leveraging different splitting techniques.</span>

            <span style='margin-left: 30px;'>- <b>Compare model answers</b> with expected correct answers.</span>

            <span style='margin-left: 30px;'>- <b>View structured outputs</b> for a clear breakdown of inputs and responses.</span>

            Selected options and results are displayed in an organized format, making it easy to experiment and assess different configurations.
            
            <i>Note: You can revisit this information anytime by clicking the reset button.</i>
            """,
            width="50%"
        )

        self.__title_widget = widgets.HTML(
            value=(
                "<h1 style='text-align: center; margin-bottom: 15px;'>"
                "VQA Approaches Exploration Form"
                "</h2>"
            )
        )

        self.__root_widget = widgets.VBox(
            children=[
                self.__title_widget,
                widgets.HBox(
                    children=[
                        self.__output_widget_manager.output_widget,
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


    def __create_general_options_layout(self) -> None:
        self.__action_type_dropdown = self.__create_dropdown(
            description="Action Type:",
            options={
                'Execute Model': 'execute_model',
                'Fetch Results from JSON': 'fetch_json'
            }
        )

        self.__country_dropdown = self.__create_dropdown(
            description='Country:',
            options={
                'Spain 🇪🇸': 'spain',
                'Brazil 🇧🇷': 'brazil',
                'Israel 🇮🇱': 'israel',
                'Japan 🇯🇵': 'japan'
            }
        )

        self.__file_type_dropdown = self.__create_dropdown(
            description='File Type:',
            options={
                'English Translation': 'english',
                'Original Language': 'local'
            }
        )

        self.__question_id_int_widget = self.__create_int_widget(
            description="Question ID:",
            initial_value=1,
            min_value=1,
            max_value=100,
            step=1
        )

        self.__vqa_strategy_type_dropdown = self.__create_dropdown(
            description="VQA Strategy Type:",
            options={
                prettify_strategy_name(vqa_strategy_type.value): vqa_strategy_type
                for vqa_strategy_type in VQAStrategyType
            }
        )

        self.__prompt_type_dropdown = self.__create_dropdown(
            description="Prompt Type:",
            options={
                prompt_type.value: prompt_type
                for prompt_type in ZeroShotPromptType
            }
        )

        self.__use_image_checkbox = self.__create_checkbox(
            description="Use Image",
            initial_value=True
        )

        self.__general_options_layout = widgets.VBox(
            children=[
                self.__action_type_dropdown,
                self.__country_dropdown,
                self.__file_type_dropdown,
                self.__question_id_int_widget,
                self.__vqa_strategy_type_dropdown,
                self.__prompt_type_dropdown,
                self.__use_image_checkbox
            ],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

    def __create_rag_options_layout(self) -> None:
        self.__create_document_splitter_options_layout()

        self.__relevant_documents_count_int_widget = self.__create_int_widget(
            description="Relevant Documents Count:",
            disabled=True,
            initial_value=1,
            min_value=1,
            max_value=5,
            step=1
        )

        self.__apply_rag_to_question_checkbox = self.__create_checkbox(
            description="Apply RAG to Question (only for RAQ Q+As)",
            initial_value=False,
            disabled=True
        )

        self.__rag_options_layout = widgets.VBox(
            children=[
                self.__relevant_documents_count_int_widget,
                self.__apply_rag_to_question_checkbox,
                self.__document_splitter_options_layout
            ],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

    def __create_document_splitter_options_layout(self) -> None:
        self.__document_splitter_type_dropdown = self.__create_dropdown(
            description="Document Splitter Type:",
            options={
                "-": "None",
                **{
                    prettify_document_splitter_name(splitter_type.value): splitter_type
                    for splitter_type in DocumentSplitterType
                }
            },
            disabled=True
        )

        self.__token_count_int_widget = self.__create_int_widget(
            description="Token Count:",
            disabled=True,
            initial_value=1,
            min_value=1,
            max_value=5,
            step=1
        )

        self.__chunk_size_int_widget = self.__create_int_widget(
            description="Chunk Size:",
            disabled=True,
            initial_value=300,
            min_value=300,
            max_value=900,
            step=300
        )

        self.__chunk_overlap_int_widget = self.__create_int_widget(
            description="Chunk Overlap:",
            disabled=True,
            initial_value=0,
            min_value=0,
            max_value=200,
            step=50
        )

        self.__add_title_checkbox = self.__create_checkbox(
            description="Use RAG Document Title",
            initial_value=False,
            disabled=True
        )

        self.__document_splitter_options_layout = widgets.Accordion(
            children=[
                widgets.VBox(
                    children=[
                        self.__document_splitter_type_dropdown,
                        self.__token_count_int_widget,
                        self.__chunk_size_int_widget,
                        self.__chunk_overlap_int_widget,
                        self.__add_title_checkbox
                    ],
                    layout=widgets.Layout(width="100%", overflow="visible")
                )
            ],
            titles=["Document Splitter Options"],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

    def __create_buttons_layout(self) -> None:
        self.__run_button = widgets.Button(description="Run")
        self.__run_button.on_click(lambda _: self.__run_form())

        self.__reset_button = widgets.Button(description="Reset")
        self.__reset_button.on_click(lambda _: self.__reset_form())

        self.__buttons_layout = widgets.HBox(
            children=[
                self.__run_button,
                self.__reset_button
            ],
            layout=widgets.Layout(
                width="100%",
                overflow="visible",
                display="flex",
                justify_content="center",
                margin="30px 0 30px 0"
            )
        )


    @staticmethod
    def __create_dropdown(
        description: str,
        options: dict,
        disabled: bool = False,
    ) -> widgets.Dropdown:
        return widgets.Dropdown(
            description=description,
            options=options,
            value=list(options.values())[0],
            layout=widgets.Layout(width="100%"),
            style={"description_width": "32%"},
            disabled=disabled
        )

    @staticmethod
    def __create_int_widget(
        description: str,
        disabled: bool = False,
        initial_value: int = 0,
        min_value: int = 0,
        max_value: int = 100,
        step: int = 1
    ) -> widgets.BoundedIntText:
        return widgets.BoundedIntText(
            description=description,
            value=initial_value,
            min=min_value,
            max=max_value,
            step=step,
            layout=widgets.Layout(width="100%"),
            style={"description_width": "32%"},
            disabled=disabled
        )

    @staticmethod
    def __create_checkbox(
        description: str,
        initial_value: bool,
        disabled: bool = False
    ) -> widgets.Checkbox:
        return widgets.Checkbox(
            description=description,
            value=initial_value,
            indent=False,
            layout=widgets.Layout(width="auto", margin="0 auto"),
            disabled=disabled
        )


    def __add_callbacks(self) -> None:
        self.__vqa_strategy_type_dropdown.observe(
            lambda change: self.__update_dependent_dropdown_values(
                change=change,
                dependent_dropdown=self.__prompt_type_dropdown,
                possible_options={
                    VQAStrategyType.ZERO_SHOT: {
                        prompt_type.value: prompt_type
                        for prompt_type in ZeroShotPromptType
                    },
                    VQAStrategyType.RAG_Q: {
                        prompt_type.value: prompt_type
                        for prompt_type in RagQPromptType
                    },
                    VQAStrategyType.RAG_Q_AS: {
                        prompt_type.value: prompt_type
                        for prompt_type in RagQPromptType
                    },
                    VQAStrategyType.RAG_IMG: {'-': '-'},
                    VQAStrategyType.RAG_DB_RERANKER: {'-': '-'}
                }
            ),
            names='value'
        )
        self.__vqa_strategy_type_dropdown.observe(
            lambda change: self.__update_dependent_widgets_state(
                dependent_widgets_config=[
                    DependentWidgetsConfig(
                        widgets=[
                            self.__relevant_documents_count_int_widget,
                            self.__document_splitter_type_dropdown
                        ],
                        enable_condition=change['new'] != VQAStrategyType.ZERO_SHOT
                    ),
                    DependentWidgetsConfig(
                        widgets=[
                            self.__apply_rag_to_question_checkbox
                        ],
                        enable_condition=change['new'] == VQAStrategyType.RAG_Q_AS
                    ),
                    DependentWidgetsConfig(
                        widgets=[
                            self.__token_count_int_widget,
                            self.__add_title_checkbox
                        ],
                        enable_condition=(
                            change['new'] != VQAStrategyType.ZERO_SHOT and
                            isinstance(self.__document_splitter_type_dropdown.value, DocumentSplitterType)
                        )
                    ),
                    DependentWidgetsConfig(
                        widgets=[
                            self.__chunk_size_int_widget,
                            self.__chunk_overlap_int_widget
                        ],
                        enable_condition=(
                            change['new'] != VQAStrategyType.ZERO_SHOT and
                            self.__document_splitter_type_dropdown.value == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER
                        )
                    )
                ]
            ),
            names='value'
        )

        self.__document_splitter_type_dropdown.observe(
            lambda change: self.__update_dependent_widgets_state(
                dependent_widgets_config=[
                    DependentWidgetsConfig(
                        widgets=[
                            self.__token_count_int_widget,
                            self.__add_title_checkbox
                        ],
                        enable_condition=isinstance(change['new'], DocumentSplitterType)
                    ),
                    DependentWidgetsConfig(
                        widgets=[
                            self.__chunk_size_int_widget,
                            self.__chunk_overlap_int_widget
                        ],
                        enable_condition=change['new'] == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER
                    )
                ]
            ),
            names='value'
        )
        self.__document_splitter_type_dropdown.observe(
            lambda change: self.__update_dependent_int_widget_values(
                change=change,
                dependent_int_widget=self.__token_count_int_widget,
                possible_values=dict(zip(
                    list(self.__document_splitter_type_dropdown.options.values()),
                    [1, 2, 5, 4]
                ))
            ),
            names='value'
        )

    @staticmethod
    def __update_dependent_dropdown_values(
        change: dict[str, Any],
        dependent_dropdown: widgets.Dropdown,
        possible_options: dict[Any, list[Any]]
    ) -> None:
        main_dropdown_selected_value = change['new']
        dependent_dropdown.options = possible_options[main_dropdown_selected_value]
        dependent_dropdown.value = list(dependent_dropdown.options.values())[0]

    def __update_dependent_widgets_state(
        self,
        dependent_widgets_config: list[DependentWidgetsConfig]
    ) -> None:
        for config in dependent_widgets_config:
            for widget in config.widgets:
                self.__update_widget_state(widget, config.enable_condition)

    @staticmethod
    def __update_widget_state(
        widget: widgets.Widget,
        enable_widget_condition: bool
    ) -> None:
        if enable_widget_condition:
            widget.disabled = False
        else:
            widget.disabled = True

    @staticmethod
    def __update_dependent_int_widget_values(
        change: dict[str, Any],
        dependent_int_widget: widgets.BoundedIntText,
        possible_values: dict[Any, int]
    ) -> None:
        main_dropdown_selected_value = change['new']
        dependent_int_widget.max = possible_values[main_dropdown_selected_value]


    def __run_form(self) -> None:
        self.__output_widget_manager.clear_content()
        self.__options_output_widget_manager.reset_content()

        dataset_name = f"{self.__country_dropdown.value}_{self.__file_type_dropdown.value}"
        row = world_med_qa_v_dataset_management.get_dataset_row_by_id(
            dataset=self.__world_med_qa_v_dataset[dataset_name],
            question_id=self.__question_id_int_widget.value
        )

        model_answer_result = self.__get_model_answer_result(row)
        self.__visualize_qa_pair_row(row, model_answer_result)
        self.__visualize_specified_options(
            output_widget_manager=self.__options_output_widget_manager,
            clear_output_content=True
        )

    def __get_model_answer_result(self, row: dict) -> ModelAnswerResult:
        if self.__action_type_dropdown.value == "execute_model":
            chosen_vqa_strategy = self.__vqa_strategy_type_dropdown.value
            model=VisualQAModel(
                visual_qa_strategy=self.__vqa_strategies[chosen_vqa_strategy],
                model_name=self.__model_name,
                country=self.__country_dropdown.value,
                file_type=self.__file_type_dropdown.value
            )
            self.__output_widget_manager.display_text_content(
                content=f"- Generating Answer for Question (ID: {row['index']}) ..."
            )
            self.__visualize_specified_options(
                output_widget_manager=self.__output_widget_manager
            )
            return model.generate_answer_from_row(
                row=row,
                possible_options=['A', 'B', 'C', 'D'],
                verbose=True,
                use_image=self.__use_image_checkbox.value,
                # logger_manager=logger_manager,
                # should_apply_rag_to_question=True
            )

        def get_widget_value(widget: widgets.Widget) -> Union[int, str]:
            if widget.disabled:
                return None
            return widget.value

        def get_doc_splitter_options() -> Optional[DocSplitterOptions]:
            if self.__document_splitter_type_dropdown.disabled:
                return None

            if self.__document_splitter_type_dropdown.value == 'None':
                return None

            return DocSplitterOptions(
                doc_splitter_type=self.__document_splitter_type_dropdown.value,
                token_count=get_widget_value(self.__token_count_int_widget),
                add_title=get_widget_value(self.__add_title_checkbox),
                chunk_size=get_widget_value(self.__chunk_size_int_widget),
                chunk_overlap=get_widget_value(self.__chunk_overlap_int_widget)
            )

        if self.__action_type_dropdown.value == "fetch_json":
            return world_med_qa_v_dataset_management.fetch_model_answer_from_json(
                evaluation_results_folder=self.__evaluation_results_folder,
                vqa_strategy_detail=VQAStrategyDetail(
                    country=self.__country_dropdown.value,
                    file_type=self.__file_type_dropdown.value,
                    use_image=self.__use_image_checkbox.value,
                    vqa_strategy_type=self.__vqa_strategy_type_dropdown.value,
                    prompt_type=self.__prompt_type_dropdown.value,
                    relevant_docs_count=get_widget_value(
                        self.__relevant_documents_count_int_widget
                    ),
                    doc_splitter_options=get_doc_splitter_options(),
                    should_apply_rag_to_question=get_widget_value(
                        self.__apply_rag_to_question_checkbox
                    )
                ),
                question_id=self.__question_id_int_widget.value
            )

        raise ValueError(f"Unexpected action type: {self.__action_type_dropdown.value}")

    def __visualize_specified_options(
        self,
        output_widget_manager: OutputWidgetManager,
        clear_output_content: bool = False
    ) -> None:
        if clear_output_content:
            output_widget_manager.clear_content()

        output_widget_manager.display_text_content(
            content="",
            title="Specified Options"
        )

        output_widget_manager.display_text_content(
            content="",
            extra_css_style="margin-left: 30px;",
            title="+ General Options"
        )
        self.__visualize_options_subset(
            output_widget_manager=output_widget_manager,
            options_widgets=self.__general_options_layout.children[1:]
        )

        if self.__vqa_strategy_type_dropdown.value != VQAStrategyType.ZERO_SHOT:
            output_widget_manager.display_text_content(
                content="",
                extra_css_style="margin-left: 30px;",
                title="+ RAG Options"
            )
            self.__visualize_options_subset(
                output_widget_manager=output_widget_manager,
                options_widgets=[
                    *self.__rag_options_layout.children[:-1],
                    *self.__document_splitter_options_layout.children[0].children
                ]
            )

    def __visualize_options_subset(
        self,
        output_widget_manager: OutputWidgetManager,
        options_widgets: list[widgets.Widget]
    ) -> None:
        widget_types_map = {
            widgets.Checkbox: lambda w: f"- {w.description}: {'✅' if w.value else '❌'}",
            widgets.Dropdown: lambda w: f"- {w.description} {w.label}",
            widgets.BoundedIntText: lambda w: f"- {w.description} {w.value}"
        }
        option_subset_rows = []

        for widget in options_widgets:
            if widget.disabled:
                continue

            widget_type = type(widget)
            if widget_type in widget_types_map:
                option_subset_rows.append(widget_types_map[widget_type](widget))
            else:
                raise ValueError(f"Unexpected widget type: {widget}")

        output_widget_manager.display_text_content(
            content="\n".join(option_subset_rows),
            extra_css_style="margin-left: 60px;"
        )

    def __visualize_qa_pair_row(
        self,
        row: dict,
        model_answer_result: Optional[ModelAnswerResult] = None,
        possible_options: Optional[list[str]] = None
    ) -> None:
        self.__output_widget_manager.clear_content()
        if possible_options is None:
            possible_options = ['A', 'B', 'C', 'D']

        # Display row id
        self.__output_widget_manager.display_text_content(
            content=str(row['index']),
            title="Question ID"
        )

        # Display question
        self.__output_widget_manager.display_text_content(
            content=row['question'],
            extra_css_style="margin-bottom: 20px;",
            title="Question"
        )

        # Display context image
        if self.__use_image_checkbox.value:
            self.__output_widget_manager.display_text_content(
                content="",
                title="Context Image"
            )
            self.__output_widget_manager.display_base64_image(base64_image=row['image'])

        # Display possible options and model answer (if provided)
        self.__display_possible_options(
            row=row,
            possible_options=possible_options,
            model_answer=model_answer_result.answer if model_answer_result else None
        )
        if model_answer_result:
            model_answer = model_answer_result.answer
            css_style = "margin-bottom: 20px 0;"

            if model_answer not in possible_options:
                model_answer = f"{model_answer} (Invalid model answer)"
                css_style = f"{css_style} color: rgb(184, 134, 11)"

            self.__output_widget_manager.display_text_content(
                content=model_answer,
                extra_css_style=css_style,
                title="Model Answer"
            )

    def __display_possible_options(
        self,
        row: dict,
        possible_options: list[str],
        model_answer: Optional[str] = None
    ) -> None:
        formatted_options = []

        def format_option(
            option_letter: str,
            option_sentence: str,
            color: str = "rgb(0, 0, 0)",
            bold: bool = False
        ) -> str:
            style = f"color: {color}; margin: 0px; padding: 0px;"
            bold_tag = "<b>" if bold else ""
            bold_end_tag = "</b>" if bold else ""

            return (
                f"<p style='{style}'>"
                f"{bold_tag}{option_letter}) {option_sentence}{bold_end_tag}"
                "</p>"
            )

        for option in possible_options:
            color = None
            bold = False

            if option == row['correct_option']:
                color = (
                    'rgb(0, 0, 255)'
                    if model_answer and model_answer not in possible_options
                    else 'rgb(0, 255, 0)'
                )
                bold = True
            elif option == model_answer:
                color = 'rgb(255, 0, 0)'
                bold = True

            formatted_options.append(
                format_option(
                    option_letter=option,
                    option_sentence=row[option],
                    color=color,
                    bold=bold
                )
            )

        formatted_options_html = "".join(formatted_options)

        self.__output_widget_manager.display_text_content(
            content=formatted_options_html,
            extra_css_style="margin-bottom: 20px;",
            title="Possible Answers"
        )

    def __reset_form(self) -> None:
        self.__output_widget_manager.reset_content()
        self.__options_output_widget_manager.reset_content()

        self.__reset_widgets_values()

    def __reset_widgets_values(self) -> None:
        self.__reset_dropdowns_value(
            dropdowns=[
                self.__action_type_dropdown,
                self.__country_dropdown,
                self.__file_type_dropdown,
                self.__vqa_strategy_type_dropdown,
                self.__prompt_type_dropdown,
                self.__document_splitter_type_dropdown
            ]
        )
        self.__reset_checkboxes_value()
        self.__reset_int_widgets_value()

    @staticmethod
    def __reset_dropdowns_value(dropdowns: list[widgets.Dropdown]) -> None:
        for dropdown in dropdowns:
            dropdown.value = list(dropdown.options.values())[0]

    def __reset_checkboxes_value(self) -> None:
        self.__use_image_checkbox.value = True
        self.__apply_rag_to_question_checkbox.value = False
        self.__add_title_checkbox.value = False

    def __reset_int_widgets_value(self) -> None:
        self.__question_id_int_widget.value = 1
        # AJUSTAR MAX

        self.__relevant_documents_count_int_widget.value = 1

        self.__token_count_int_widget.value = 1

        self.__chunk_size_int_widget.value = 300
        self.__chunk_overlap_int_widget.value = 0
