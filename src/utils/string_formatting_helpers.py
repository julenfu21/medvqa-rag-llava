def to_snake_case_strategy_name(strategy_name: str) -> str:
    return (
        strategy_name.
        lower()
        .replace("-", "_")
        .replace("+", "_")
        .replace(" ", "_")
    )


def prettify_strategy_name(strategy_name: str) -> str:
    snake_case_to_pretty_name = {
        'zero_shot': 'Zero-Shot',
        'rag_q': 'RAG Q',
        'rag_q_as': 'RAG Q+As',
        'rag_img': 'RAG IMG',
        'rag_db_reranker': 'RAG DB-Reranker'
    }

    return snake_case_to_pretty_name[strategy_name]
