from itertools import chain

from src.utils.data_definitions import VQAStrategyDetail
from src.utils.types_aliases import PromptType
from src.utils.enums import DocumentSplitterType, VQAStrategyType


def build_evaluate_vqa_model_python_code(
    vqa_strategy_detail: VQAStrategyDetail
) -> str:
    # Import Statements
    full_prompt_type_name, prompt_type_enum_class = _get_prompt_type_class_info(
        prompt_type=vqa_strategy_detail.prompt_type
    )
    import_statements = [
        "from pathlib import Path",
        "",
        "from src.utils.dataset_helpers.world_med_qa_v.dataset_management import load_vqa_dataset",
        f"from src.utils.enums import {prompt_type_enum_class}"
    ]

    doc_splitter_options = vqa_strategy_detail.doc_splitter_options
    if doc_splitter_options:
        document_splitter_class_name = _get_document_splitter_class_name(
            document_splitter_type=doc_splitter_options.doc_splitter_type
        )
        import_statements.append((
            f"from src.utils.text_splitters.{_pascal_to_snake_case(document_splitter_class_name)} "
            f"import {document_splitter_class_name}"
        ))

    vqa_strategy_type = vqa_strategy_detail.vqa_strategy_type
    vqa_strategy_class_name = _get_vqa_strategy_class_name(vqa_strategy_type=vqa_strategy_type)
    import_statements.extend([
        "from src.visual_qa_model import VisualQAModel",
        (
            f"from src.visual_qa_strategies.{_pascal_to_snake_case(vqa_strategy_class_name)} "
            f"import {vqa_strategy_class_name}"
        )
    ])


    # Constants
    constants = [
        "DATASET_DIR = Path('data/WorldMedQA-V')",
        "OLLAMA_MODEL_NAME = 'llava'",
        "RESULTS_DIR = Path('evaluation_results')"
    ]

    if vqa_strategy_type != VQAStrategyType.ZERO_SHOT:
        constants.extend([
            "",
            "INDEX_DIR = Path('data/WikiMed/indexed_db')",
            "INDEX_NAME = 'Wikimed+S-PubMedBert-MS-MARCO-FullTexts'",
            "EMBEDDING_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO'",
            f"RELEVANT_DOCS_COUNT = {vqa_strategy_detail.relevant_docs_count}"
        ])


    # Dataset Load
    country = vqa_strategy_detail.country
    file_type = vqa_strategy_detail.file_type
    dataset_load = [
        "world_med_qa_v_dataset = load_vqa_dataset(",
        _get_indented_text(content="data_path=DATASET_DIR,", level=1),
        _get_indented_text(content=f"country='{country}',", level=1),
        _get_indented_text(content=f"file_type='{file_type}'", level=1),
        ")"
    ]


    # Model Load
    model_load = [
        "llava_model = VisualQAModel(",
        _get_indented_text(
            content=f"visual_qa_strategy={vqa_strategy_class_name}(",
            level=1
        ),
        _get_indented_text(
            content=f"prompt_type={full_prompt_type_name},",
            level=2
        )
    ]

    if vqa_strategy_type in (VQAStrategyType.RAG_Q, VQAStrategyType.RAG_Q_AS):
        model_load.extend([
            _get_indented_text(content="index_dir=INDEX_DIR,", level=2),
            _get_indented_text(content="index_name=INDEX_NAME,", level=2),
            _get_indented_text(content="embedding_model_name=EMBEDDING_MODEL_NAME,", level=2),
            _get_indented_text(content="relevant_docs_count=RELEVANT_DOCS_COUNT", level=2)
        ])

    model_load.extend([
        _get_indented_text(content="),", level=1),
        _get_indented_text(content="model_name=OLLAMA_MODEL_NAME,", level=1),
        _get_indented_text(content=f"country='{country}',", level=1),
        _get_indented_text(content=f"file_type='{file_type}'", level=1),
        ")",
    ])


    # Model Evaluation
    model_evaluation = [
        "llava_model.evaluate(",
        _get_indented_text(content="dataset=world_med_qa_v_dataset,", level=1),
        _get_indented_text(content="results_path=RESULTS_DIR,", level=1),
        _get_indented_text(content=f"use_image={vqa_strategy_detail.use_image},", level=1)
    ]

    if doc_splitter_options:
        model_evaluation.append(_get_indented_text(
            content=f"doc_splitter={document_splitter_class_name}(",
            level=1
        ))
        if doc_splitter_options.token_count:
            model_evaluation.append(_get_indented_text(
                content=f"token_count={doc_splitter_options.token_count},",
                level=2
            ))
        if doc_splitter_options.add_title:
            model_evaluation.append(_get_indented_text(
                content=f"add_title={doc_splitter_options.add_title},",
                level=2
            ))
        if doc_splitter_options.chunk_size:
            model_evaluation.append(_get_indented_text(
                content=f"chunk_size={doc_splitter_options.chunk_size},",
                level=2
            ))
        if doc_splitter_options.chunk_overlap:
            model_evaluation.append(_get_indented_text(
                content=f"chunk_size={doc_splitter_options.chunk_overlap},",
                level=2
            ))
        model_evaluation.append(_get_indented_text(
            content="),",
            level=1
        ))

    should_apply_rag_to_question = vqa_strategy_detail.should_apply_rag_to_question
    if vqa_strategy_type == VQAStrategyType.RAG_Q_AS:
        model_evaluation.append(_get_indented_text(
            content=f"should_apply_rag_to_question={should_apply_rag_to_question}",
            level=1
        ))

    model_evaluation.append(")")


    # Join all Python Code blocks into a Single String
    code_lines = list(chain.from_iterable([
        import_statements,
        "\n",
        constants,
        "\n",
        dataset_load,
        "\n",
        model_load,
        "\n",
        model_evaluation
    ]))

    return "\n".join(code_lines)


def _get_prompt_type_class_info(prompt_type: PromptType) -> tuple[str, str]:
    short_vqa_strategy_type, sub_prompt_type = prompt_type.value.split('_')
    short_vqa_strategy_to_enum_class_name = {
        'zs': 'ZeroShotPromptType',
        'rq': 'RagQPromptType'
    }
    prompt_type_enum_class = short_vqa_strategy_to_enum_class_name[short_vqa_strategy_type]
    full_prompt_type_name = f"{prompt_type_enum_class}.{sub_prompt_type.upper()}"

    return full_prompt_type_name, prompt_type_enum_class

def _pascal_to_snake_case(name: str):
    return ''.join(
        char if char.islower() else f'_{char.lower()}' for char in name
    ).lstrip('_').replace('v_q_a', 'vqa')

def _get_document_splitter_class_name(document_splitter_type: DocumentSplitterType) -> str:
    match document_splitter_type:
        case DocumentSplitterType.NO_SPLITTER:
            return 'NoSplitter'
        case DocumentSplitterType.RECURSIVE_CHARACTER_SPLITTER:
            return 'RecursiveCharacterSplitter'
        case DocumentSplitterType.SPACY_SENTENCE_SPLITTER:
            return 'SpacySentenceSplitter'
        case DocumentSplitterType.PARAGRAPH_SPLITTER:
            return 'ParagraphSplitter'

def _get_vqa_strategy_class_name(vqa_strategy_type: VQAStrategyType) -> str:
    match vqa_strategy_type:
        case VQAStrategyType.ZERO_SHOT:
            return 'ZeroShotVQAStrategy'
        case VQAStrategyType.RAG_Q:
            return 'RagQVQAStrategy'
        case VQAStrategyType.RAG_Q_AS:
            return 'RagQAsVQAStrategy'

    raise TypeError("Unhandled VQA strategy type")

def _get_indented_text(content: str, level: int = 1, px_per_level: int = 30) -> str:
    margin = px_per_level * level
    return f"<span style='margin-left: {margin}px;'>{content}</span>"
