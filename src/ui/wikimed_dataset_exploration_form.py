from pathlib import Path
from typing import Optional

import ipywidgets as widgets
import pandas as pd
from datasets import Dataset
from langchain_core.documents import Document

import src.utils.dataset_helpers.wikimed.dataset_management as wikimed_dataset_management
from src.ui.base_interactive_form import BaseInteractiveForm
from src.ui.utils.data_definitions import DependentWidgetsConfig
from src.ui.utils.display_utils import visualize_wikimed_document
from src.ui.utils.widget_utils import (
    get_widget_value,
    update_dependent_int_widget_values_from_change,
    update_dependent_widgets_state
)
from src.ui.widgets.widget_factory import (
    create_checkbox,
    create_dropdown,
    create_int_widget,
    create_text_field
)
from src.utils.data_definitions import DocSplitterOptions
from src.utils.enums import DocumentSplitterType
from src.utils.string_formatting_helpers import prettify_document_splitter_name
from src.utils.text_splitters.base_splitter import BaseSplitter
from src.utils.text_splitters.no_splitter import NoSplitter
from src.utils.text_splitters.paragraph_splitter import ParagraphSplitter
from src.utils.text_splitters.recursive_character_splitter import RecursiveCharacterSplitter
from src.utils.text_splitters.spacy_sentence_splitter import SpacySentenceSplitter


class WikimedDatasetExplorationForm(BaseInteractiveForm):

    def __init__(
        self,
        dataset_path: Path,
        dataset_metadata: pd.DataFrame,
        dataset: dict[str, Dataset] = None
    ) -> None:
        self.__dataset_path = dataset_path
        self.__dataset_metadata = dataset_metadata

        # Options Layout elements
        self.__options_layout = None
        self.__doc_title_text_widget = None
        self.__document_splitter_type_dropdown = None
        self.__token_count_int_widget = None
        self.__chunk_size_int_widget = None
        self.__chunk_overlap_int_widget = None
        self.__add_title_checkbox = None

        # Create Form
        super().__init__(
            form_title="WikiMed Dataset Exploration Form",
            dataset=dataset,
            options_layout_width=35
        )


    def _get_initial_output_widget_text(self) -> str:
        return """
        This interactive form lets you explore the <b>WikiMed Dataset</b> by providing a document title and adjusting processing options. You can:

        <span style='margin-left: 30px;'>- <b>Search documents</b> by entering a specific title from the WikiMed Dataset.</span>

        <span style='margin-left: 30px;'>- <b>Apply document splitting</b> techniques to see how the content would be segmented before being passed to the model.</span>

        The processed document view helps you understand how different splitting strategies affect model input structure.

        <i>Note: You can revisit this information anytime by clicking the reset button.</i>
        """


    def _create_options_accordion(self) -> widgets.Accordion:
        self.__options_layout = self.__create_options_layout()

        return widgets.Accordion(
            children=[
                self.__options_layout
            ],
            titles=[
                "Options"
            ],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

    def __create_options_layout(self) -> widgets.VBox:
        self.__doc_title_text_widget = create_text_field(
            description="Document Title:",
            placeholder="Enter the title of a document"
        )
        self.__document_splitter_type_dropdown = create_dropdown(
            description="Doc. Splitter Type:",
            options={
                prettify_document_splitter_name(splitter_type.value): splitter_type
                for splitter_type in DocumentSplitterType
            },
            disabled=False
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
            initial_value=True,
            disabled=False
        )

        return widgets.VBox(
            children=[
                self.__doc_title_text_widget.widget,
                self.__document_splitter_type_dropdown.widget,
                self.__token_count_int_widget.widget,
                self.__chunk_size_int_widget.widget,
                self.__chunk_overlap_int_widget.widget,
                self.__add_title_checkbox.widget
            ],
            layout=widgets.Layout(width="100%", overflow="visible")
        )

    def _get_initial_options_output_widget_text(self) -> str:
        return ""


    def _add_callbacks(self) -> None:
        self.__document_splitter_type_dropdown.widget.observe(
            lambda change: update_dependent_widgets_state(
                dependent_widgets_config=[
                    DependentWidgetsConfig(
                        widgets=[
                            self.__token_count_int_widget.widget
                        ],
                        enable_condition=change['new'] != DocumentSplitterType.NO_SPLITTER
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
            lambda change: update_dependent_int_widget_values_from_change(
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

        document_title = self.__doc_title_text_widget.widget.value
        try:
            row = wikimed_dataset_management.get_dataset_row_by_doc_title(
                dataset_path=self.__dataset_path,
                dataset_metadata=self.__dataset_metadata,
                doc_title=document_title
            )
        except ValueError:
            self._output_widget_manager.display_text_content(
                content=f"""
                ⚠️ Unable to find a document with title: {document_title} ⚠️
                
                Make sure a document with the title entered exists.
                """,
                extra_css_style="color: #b71c1c; font-weight: bold;"
            )
        else:
            doc_splitter_options = self.__get_doc_splitter_options()
            document_splitter = self.__get_doc_splitter_from_options(doc_splitter_options)
            document = Document(page_content=row['text'])

            if document_splitter:
                document_text = document_splitter.split_documents(documents=[document])[0]
            else:
                document_text = document.page_content

            visualize_wikimed_document(
                output_widget_manager=self._output_widget_manager,
                row=row,
                document_text=document_text
            )

    def __get_doc_splitter_options(self) -> DocSplitterOptions:
        return DocSplitterOptions(
            doc_splitter_type=self.__document_splitter_type_dropdown.widget.value,
            token_count=get_widget_value(self.__token_count_int_widget.widget),
            add_title=get_widget_value(self.__add_title_checkbox.widget),
            chunk_size=get_widget_value(self.__chunk_size_int_widget.widget),
            chunk_overlap=get_widget_value(self.__chunk_overlap_int_widget.widget)
        )

    @staticmethod
    def __get_doc_splitter_from_options(doc_splitter_options: DocSplitterOptions) -> BaseSplitter:
        match doc_splitter_options.doc_splitter_type:
            case DocumentSplitterType.NO_SPLITTER:
                return NoSplitter(
                    token_count=doc_splitter_options.token_count,
                    add_title=doc_splitter_options.add_title
                )
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


    def _reset_form_specific_widgets(self) -> None:
        super()._reset_widgets(
            widget_wrappers=[
                self.__doc_title_text_widget,
                self.__document_splitter_type_dropdown,
                self.__token_count_int_widget,
                self.__chunk_size_int_widget,
                self.__chunk_overlap_int_widget,
                self.__add_title_checkbox
            ]
        )
