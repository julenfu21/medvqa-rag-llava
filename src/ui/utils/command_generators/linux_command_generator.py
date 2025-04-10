from src.utils.data_definitions import VQAStrategyDetail


def build_evaluate_vqa_model_linux_command(
    vqa_strategy_detail: VQAStrategyDetail
) -> str:
    command_elements = [
        "python scripts/evaluate_vqa_model.py \\",
        f"> --country={vqa_strategy_detail.country} \\",
        f"> --file_type={vqa_strategy_detail.file_type} \\"
    ]

    if not vqa_strategy_detail.use_image:
        command_elements.append("> --no-image \\")

    command_elements.extend([
        f"> --vqa_strategy={vqa_strategy_detail.vqa_strategy_type} \\",
        f"> --prompt_type={vqa_strategy_detail.prompt_type} \\"
    ])

    relevant_docs_count = vqa_strategy_detail.relevant_docs_count
    if relevant_docs_count:
        command_elements.append(f"> --relevant_docs_count={relevant_docs_count} \\")

    doc_splitter_options = vqa_strategy_detail.doc_splitter_options
    if doc_splitter_options:
        command_elements.append(f"> --doc_splitter={doc_splitter_options.doc_splitter_type} \\")
        if doc_splitter_options.token_count:
            command_elements.append(f"> --token_count={doc_splitter_options.token_count} \\")
        if doc_splitter_options.add_title:
            command_elements.append("> --add_title \\")
        if doc_splitter_options.chunk_size:
            command_elements.append(f"> --chunk_size={doc_splitter_options.chunk_size} \\")
        if doc_splitter_options.chunk_overlap:
            command_elements.append(f"> --chunk_overlap={doc_splitter_options.chunk_overlap} \\")

    if vqa_strategy_detail.should_apply_rag_to_question:
        command_elements.append("> --should_apply_rag_to_question \\")

    command_elements.append("> -v")

    return "\n".join(command_elements)
