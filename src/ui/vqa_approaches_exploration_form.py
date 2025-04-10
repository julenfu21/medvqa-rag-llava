from pathlib import Path
from typing import Optional

import ipywidgets as widgets
from datasets import Dataset

import src.utils.dataset_helpers.world_med_qa_v.dataset_management as world_med_qa_v_dataset_management
from src.ui.base_interactive_form import BaseInteractiveForm
from src.ui.utils.command_generators.linux_command_generator import build_evaluate_vqa_model_linux_command
from src.ui.utils.data_definitions import DependentWidgetsConfig
from src.ui.utils.display_utils import visualize_options_subset, visualize_qa_pair_row
from src.ui.utils.widget_utils import (
    get_widget_value,
    update_dependent_dropdown_values,
    update_dependent_int_widget_values,
    update_dependent_widgets_state
)
from src.ui.widgets.clipboard_copy_widget import ClipboardCopyWidget
from src.ui.widgets.output_widget_manager import OutputWidgetManager
from src.ui.widgets.widget_factory import (
    create_checkbox,
    create_dropdown,
    create_int_widget
)
from src.ui.utils.command_generators.python_code_generator import build_evaluate_vqa_model_python_code
from src.utils.data_definitions import (
    DocSplitterOptions,
    ModelAnswerResult,
    VQAStrategyDetail
)
from src.utils.enums import (
    CommandType,
    DocumentSplitterType,
    OutputFileType,
    RagQPromptType,
    VQAStrategyType,
    ZeroShotPromptType
)
from src.utils.logger import LoggerManager
from src.utils.string_formatting_helpers import (
    prettify_document_splitter_name,
    prettify_strategy_name
)
from src.utils.text_splitters.paragraph_splitter import ParagraphSplitter
from src.utils.text_splitters.recursive_character_splitter import RecursiveCharacterSplitter
from src.utils.text_splitters.spacy_sentence_splitter import SpacySentenceSplitter
from src.visual_qa_model import VisualQAModel
from src.visual_qa_strategies.base_rag_strategy import BaseRagVQAStrategy
from src.visual_qa_strategies.base_vqa_strategy import BaseVQAStrategy


class VQAApproachesExplorationForm(BaseInteractiveForm):

    def __init__(
        self,
        dataset: dict[str, Dataset],
        model_name: str,
        vqa_strategies: dict[VQAStrategyType, BaseVQAStrategy],
        evaluation_results_folder: Path,
        logger_manager: LoggerManager
    ) -> None:
        self.__model_name = model_name
        self.__vqa_strategies = vqa_strategies
        self.__evaluation_results_folder = evaluation_results_folder
        self.__logger_manager = logger_manager

        # Main 'Options Layout' elements
        self.__general_options_layout = None
        self.__rag_options_layout = None

        # General Options Widgets
        self.__action_type_dropdown = None
        self.__country_dropdown = None
        self.__file_type_dropdown = None
        self.__question_id_int_widget = None
        self.__vqa_strategy_type_dropdown = None
        self.__prompt_type_dropdown = None
        self.__use_image_checkbox = None
        self.__create_log_file_checkbox = None

        # RAG Options Widgets
        self.__relevant_documents_count_int_widget = None
        self.__apply_rag_to_question_checkbox = None

        # Document Splitter Options Widgets
        self.__document_splitter_options_layout = None
        self.__document_splitter_type_dropdown = None
        self.__token_count_int_widget = None
        self.__chunk_size_int_widget = None
        self.__chunk_overlap_int_widget = None
        self.__add_title_checkbox = None

        # Create Form
        super().__init__(form_title="VQA Approaches Exploration Form", dataset=dataset)


    def _get_initial_output_widget_text(self) -> str:
        return """
        This interactive form lets you configure and explore different <b>Visual Question Answering (VQA) strategies</b> by adjusting various options. You can:

        <span style='margin-left: 30px;'>- Set <b>key inputs</b>, like country, file type, question ID, and VQA strategy.</span>

        <span style='margin-left: 30px;'>- <b>Customize document processing</b>, leveraging different splitting techniques.</span>

        <span style='margin-left: 30px;'>- <b>Compare model answers</b> with expected correct answers.</span>

        <span style='margin-left: 30px;'>- <b>View structured outputs</b> for a clear breakdown of inputs and responses.</span>

        <span style='margin-left: 30px;'>- <b>Explore message logs</b> to inspect the messages exchanged during processing.</span>

        Selected options and results are displayed in an organized format, making it easy to experiment and assess different configurations.
        
        <i>Note: You can revisit this information anytime by clicking the reset button.</i>
        """


    def _get_initial_options_output_widget_text(self) -> str:
        return """
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
        """


    def _create_options_accordion(self) -> widgets.Accordion:
        self.__general_options_layout = self.__create_general_options_layout()
        self.__rag_options_layout = self.__create_rag_options_layout()

        return widgets.Accordion(
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

    def __create_general_options_layout(self) -> widgets.VBox:
        self.__action_type_dropdown = create_dropdown(
            description="Action Type:",
            options={
                'Execute Model': 'execute_model',
                'Fetch Results from JSON': 'fetch_json'
            }
        )
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
            max_value=100,
            step=1
        )
        self.__vqa_strategy_type_dropdown = create_dropdown(
            description="VQA Strategy Type:",
            options={
                prettify_strategy_name(vqa_strategy_type.value): vqa_strategy_type
                for vqa_strategy_type in VQAStrategyType
            }
        )
        self.__prompt_type_dropdown = create_dropdown(
            description="Prompt Type:",
            options={
                prompt_type.value: prompt_type
                for prompt_type in ZeroShotPromptType
            }
        )
        self.__use_image_checkbox = create_checkbox(
            description="Use Image",
            initial_value=True
        )
        self.__create_log_file_checkbox = create_checkbox(
            description="Create Log File",
            initial_value=False
        )

        return widgets.VBox(
            children=[
                self.__action_type_dropdown.widget,
                self.__country_dropdown.widget,
                self.__file_type_dropdown.widget,
                self.__question_id_int_widget.widget,
                self.__vqa_strategy_type_dropdown.widget,
                self.__prompt_type_dropdown.widget,
                widgets.HBox(
                    children=[
                        self.__use_image_checkbox.widget,
                        self.__create_log_file_checkbox.widget
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

    def __create_rag_options_layout(self) -> widgets.VBox:
        self.__relevant_documents_count_int_widget = create_int_widget(
            description="Relevant Documents Count:",
            disabled=True,
            initial_value=1,
            min_value=1,
            max_value=5,
            step=1
        )
        self.__apply_rag_to_question_checkbox = create_checkbox(
            description="Apply RAG to Question (only for RAQ Q+As)",
            initial_value=False,
            disabled=True
        )
        self.__document_splitter_options_layout = self.__create_document_splitter_options_layout()

        return widgets.VBox(
            children=[
                self.__relevant_documents_count_int_widget.widget,
                self.__apply_rag_to_question_checkbox.widget,
                self.__document_splitter_options_layout
            ],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

    def __create_document_splitter_options_layout(self) -> widgets.Accordion:
        self.__document_splitter_type_dropdown = create_dropdown(
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
        self.__token_count_int_widget = create_int_widget(
            description="Token Count:",
            disabled=True,
            initial_value=1,
            min_value=1,
            max_value=5,
            step=1
        )
        self.__chunk_size_int_widget = create_int_widget(
            description="Chunk Size:",
            disabled=True,
            initial_value=300,
            min_value=300,
            max_value=900,
            step=300
        )
        self.__chunk_overlap_int_widget = create_int_widget(
            description="Chunk Overlap:",
            disabled=True,
            initial_value=0,
            min_value=0,
            max_value=200,
            step=50
        )
        self.__add_title_checkbox = create_checkbox(
            description="Use RAG Document Title",
            initial_value=False,
            disabled=True
        )

        return widgets.Accordion(
            children=[
                widgets.VBox(
                    children=[
                        self.__document_splitter_type_dropdown.widget,
                        self.__token_count_int_widget.widget,
                        self.__chunk_size_int_widget.widget,
                        self.__chunk_overlap_int_widget.widget,
                        self.__add_title_checkbox.widget
                    ],
                    layout=widgets.Layout(width="100%", overflow="visible")
                )
            ],
            titles=["Document Splitter Options"],
            layout=widgets.Layout(width="100%", overflow="visible")
        )


    def _add_callbacks(self) -> None:
        self.__action_type_dropdown.widget.observe(
            lambda change: update_dependent_widgets_state(
                dependent_widgets_config=[
                    DependentWidgetsConfig(
                        widgets=[self.__create_log_file_checkbox.widget],
                        enable_condition=change['new'] == 'execute_model'
                    )
                ]
            ),
            names='value'
        )

        self.__vqa_strategy_type_dropdown.widget.observe(
            lambda change: update_dependent_dropdown_values(
                change=change,
                dependent_dropdown=self.__prompt_type_dropdown.widget,
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
        self.__vqa_strategy_type_dropdown.widget.observe(
            lambda change: update_dependent_widgets_state(
                dependent_widgets_config=[
                    DependentWidgetsConfig(
                        widgets=[
                            self.__relevant_documents_count_int_widget.widget,
                            self.__document_splitter_type_dropdown.widget
                        ],
                        enable_condition=change['new'] != VQAStrategyType.ZERO_SHOT
                    ),
                    DependentWidgetsConfig(
                        widgets=[
                            self.__apply_rag_to_question_checkbox.widget
                        ],
                        enable_condition=change['new'] == VQAStrategyType.RAG_Q_AS
                    ),
                    DependentWidgetsConfig(
                        widgets=[
                            self.__token_count_int_widget.widget,
                            self.__add_title_checkbox.widget
                        ],
                        enable_condition=(
                            change['new'] != VQAStrategyType.ZERO_SHOT and
                            isinstance(self.__document_splitter_type_dropdown.widget.value, DocumentSplitterType)
                        )
                    ),
                    DependentWidgetsConfig(
                        widgets=[
                            self.__chunk_size_int_widget.widget,
                            self.__chunk_overlap_int_widget.widget
                        ],
                        enable_condition=(
                            change['new'] != VQAStrategyType.ZERO_SHOT and
                            self.__document_splitter_type_dropdown.widget.value == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER
                        )
                    )
                ]
            ),
            names='value'
        )

        self.__document_splitter_type_dropdown.widget.observe(
            lambda change: update_dependent_widgets_state(
                dependent_widgets_config=[
                    DependentWidgetsConfig(
                        widgets=[
                            self.__token_count_int_widget.widget,
                            self.__add_title_checkbox.widget
                        ],
                        enable_condition=isinstance(change['new'], DocumentSplitterType)
                    ),
                    DependentWidgetsConfig(
                        widgets=[
                            self.__chunk_size_int_widget.widget,
                            self.__chunk_overlap_int_widget.widget
                        ],
                        enable_condition=change['new'] == DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER
                    )
                ]
            ),
            names='value'
        )
        self.__document_splitter_type_dropdown.widget.observe(
            lambda change: update_dependent_int_widget_values(
                change=change,
                dependent_int_widget=self.__token_count_int_widget.widget,
                possible_values=dict(zip(
                    list(self.__document_splitter_type_dropdown.widget.options.values()),
                    [1, 2, 5, 4]
                ))
            ),
            names='value'
        )


    def _run_form(self) -> None:
        self._output_widget_manager.clear_content()
        self._options_output_widget_manager.clear_content()

        dataset_name = (
            f"{self.__country_dropdown.widget.value}_"
            f"{self.__file_type_dropdown.widget.value}"
        )
        row = world_med_qa_v_dataset_management.get_dataset_row_by_id(
            dataset=self._dataset[dataset_name],
            question_id=self.__question_id_int_widget.widget.value
        )

        try:
            model_answer_result = self.__get_model_answer_result(row)
        except FileNotFoundError:
            self.__visualize_evaluation_commands()
            model_answer_result = None
        finally:
            self.__visualize_specified_options(
                output_widget_manager=self._options_output_widget_manager
            )

        if model_answer_result:
            visualize_qa_pair_row(
                output_widget_manager=self._output_widget_manager,
                display_image=self.__use_image_checkbox.widget.value,
                row=row,
                model_answer_result=model_answer_result
            )

    def __get_model_answer_result(self, row: dict) -> ModelAnswerResult:
        vqa_strategy_detail = self.__build_vqa_strategy_detail()

        if self.__action_type_dropdown.widget.value == "execute_model":
            if self.__create_log_file_checkbox.widget.value:
                self.__create_log_file(vqa_strategy_detail=vqa_strategy_detail)

            vqa_strategy = self.__vqa_strategies[vqa_strategy_detail.vqa_strategy_type]
            if isinstance(vqa_strategy, BaseRagVQAStrategy):
                vqa_strategy.relevant_docs_count = (
                    self.__relevant_documents_count_int_widget.widget.value
                )
            model=VisualQAModel(
                visual_qa_strategy=vqa_strategy,
                model_name=self.__model_name,
                country=vqa_strategy_detail.country,
                file_type=vqa_strategy_detail.file_type
            )
            self._output_widget_manager.display_text_content(
                content=f"- Generating Answer for Question (ID: {row['index']}) ..."
            )
            self.__visualize_specified_options(
                output_widget_manager=self._output_widget_manager
            )

            return model.generate_answer_from_row(
                row=row,
                possible_options=['A', 'B', 'C', 'D'],
                verbose=True,
                use_image=vqa_strategy_detail.use_image,
                logger_manager=(
                    self.__logger_manager if self.__create_log_file_checkbox.widget.value
                    else None
                ),
                doc_splitter=self.__get_doc_splitter_from_options(
                    doc_splitter_options=vqa_strategy_detail.doc_splitter_options
                ),
                **(
                    {"should_apply_rag_to_question": vqa_strategy_detail.should_apply_rag_to_question}
                    if vqa_strategy_detail.vqa_strategy_type == VQAStrategyType.RAG_Q_AS
                    else {}
                )
            )

        if self.__action_type_dropdown.widget.value == "fetch_json":
            return world_med_qa_v_dataset_management.fetch_model_answer_from_json(
                evaluation_results_folder=self.__evaluation_results_folder,
                vqa_strategy_detail=vqa_strategy_detail,
                question_id=self.__question_id_int_widget.widget.value
            )

        raise ValueError(f"Unexpected action type: {self.__action_type_dropdown.widget.value}")

    def __build_vqa_strategy_detail(self) -> VQAStrategyDetail:
        return VQAStrategyDetail(
            country=self.__country_dropdown.widget.value,
            file_type=self.__file_type_dropdown.widget.value,
            use_image=self.__use_image_checkbox.widget.value,
            vqa_strategy_type=self.__vqa_strategy_type_dropdown.widget.value,
            prompt_type=self.__prompt_type_dropdown.widget.value,
            relevant_docs_count=get_widget_value(
                self.__relevant_documents_count_int_widget.widget
            ),
            doc_splitter_options=self.__get_doc_splitter_options(),
            should_apply_rag_to_question=get_widget_value(
                self.__apply_rag_to_question_checkbox.widget
            )
        )

    def __get_doc_splitter_options(self) -> Optional[DocSplitterOptions]:
        if self.__document_splitter_type_dropdown.widget.disabled:
            return None

        if self.__document_splitter_type_dropdown.widget.value == 'None':
            return None

        return DocSplitterOptions(
            doc_splitter_type=self.__document_splitter_type_dropdown.widget.value,
            token_count=get_widget_value(self.__token_count_int_widget.widget),
            add_title=get_widget_value(self.__add_title_checkbox.widget),
            chunk_size=get_widget_value(self.__chunk_size_int_widget.widget),
            chunk_overlap=get_widget_value(self.__chunk_overlap_int_widget.widget)
        )

    def __create_log_file(self, vqa_strategy_detail: VQAStrategyDetail) -> None:
        log_filepath = vqa_strategy_detail.generate_output_filepath(
            root_folder=self.__logger_manager.log_save_directory,
            output_file_type=OutputFileType.LOG_FILE
        )
        self.__logger_manager.create_new_log_file(log_filepath=log_filepath)
        self._output_widget_manager.display_text_content(
            content=f"""
            - New Message Log Created: 
            <span style='margin-left: 30px;'>+ Path: {self.__logger_manager.log_filepath.parent}</span>
            <span style='margin-left: 30px;'>+ Log Filename: {self.__logger_manager.log_filepath.name}</span>
            """
        )

    @staticmethod
    def __get_doc_splitter_from_options(doc_splitter_options: DocSplitterOptions) -> None:
        if not doc_splitter_options:
            return None

        match doc_splitter_options.doc_splitter_type:
            case DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER:
                return RecursiveCharacterSplitter(
                    token_count=doc_splitter_options.token_count,
                    chunk_size=doc_splitter_options.chunk_size,
                    chunk_overlap=doc_splitter_options.chunk_overlap,
                    add_title=doc_splitter_options.add_title
                )
            case DocumentSplitterType.SPACY_SENTENCE_SPLITTER:
                return SpacySentenceSplitter(
                    token_count=doc_splitter_options.token_count,
                    add_title=doc_splitter_options.add_title
                )
            case DocumentSplitterType.PARAGRAPH_SPLITTER:
                return ParagraphSplitter(
                    token_count=doc_splitter_options.token_count,
                    add_title=doc_splitter_options.add_title
                )

    def __visualize_evaluation_commands(self) -> None:
        self._output_widget_manager.display_text_content(
            content=(
                "You have not evaluated the LLaVA model with these options. This can be done "
                "with one of the following commands:"
            ),
            extra_css_style="margin-bottom: 40px;"
        )

        linux_command_clipboard_copy_widget = ClipboardCopyWidget(
            header="Linux Command 🖥️🐧",
            command_type=CommandType.LINUX_COMMAND,
            text_content=build_evaluate_vqa_model_linux_command(
                vqa_strategy_detail=self.__build_vqa_strategy_detail()
            )
        )
        python_command_clipboard_copy_widget = ClipboardCopyWidget(
            header="Python Code 🐍💻",
            command_type=CommandType.PYTHON_CODE,
            text_content=build_evaluate_vqa_model_python_code(
                vqa_strategy_detail=self.__build_vqa_strategy_detail()
            )
        )

        self._output_widget_manager.display_widget(
            widget=linux_command_clipboard_copy_widget.root_widget
        )
        self._output_widget_manager.display_widget(
            widget=python_command_clipboard_copy_widget.root_widget
        )

    def __visualize_specified_options(
        self,
        output_widget_manager: OutputWidgetManager
    ) -> None:
        output_widget_manager.display_text_content(
            content="",
            title="Specified Options"
        )

        output_widget_manager.display_text_content(
            content="",
            extra_css_style="margin-left: 30px;",
            title="+ General Options"
        )
        visualize_options_subset(
            output_widget_manager=output_widget_manager,
            options_widgets=[
                self.__country_dropdown.widget,
                self.__file_type_dropdown.widget,
                self.__question_id_int_widget.widget,
                self.__vqa_strategy_type_dropdown.widget,
                self.__prompt_type_dropdown.widget,
                self.__use_image_checkbox.widget
            ]
        )

        if self.__vqa_strategy_type_dropdown.widget.value != VQAStrategyType.ZERO_SHOT:
            output_widget_manager.display_text_content(
                content="",
                extra_css_style="margin-left: 30px;",
                title="+ RAG Options"
            )
            visualize_options_subset(
                output_widget_manager=output_widget_manager,
                options_widgets=[
                    self.__relevant_documents_count_int_widget.widget,
                    self.__apply_rag_to_question_checkbox.widget,
                    self.__document_splitter_type_dropdown.widget,
                    self.__token_count_int_widget.widget,
                    self.__chunk_size_int_widget.widget,
                    self.__chunk_overlap_int_widget.widget,
                    self.__add_title_checkbox.widget
                ]
            )


    def _reset_form_specific_widgets(self) -> None:
        super()._reset_widgets(
            widget_wrappers=[
                self.__action_type_dropdown,
                self.__country_dropdown,
                self.__file_type_dropdown,
                self.__question_id_int_widget,
                self.__vqa_strategy_type_dropdown,
                self.__prompt_type_dropdown,
                self.__use_image_checkbox,
                self.__create_log_file_checkbox,
                self.__document_splitter_type_dropdown,
                self.__relevant_documents_count_int_widget,
                self.__apply_rag_to_question_checkbox,
                self.__document_splitter_type_dropdown,
                self.__token_count_int_widget,
                self.__chunk_size_int_widget,
                self.__chunk_overlap_int_widget,
                self.__add_title_checkbox
            ]
        )
