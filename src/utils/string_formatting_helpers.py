def format_vqa_strategy_name(strategy_name: str) -> str:
    return (
        strategy_name.
        lower()
        .replace("-", "_")
        .replace("+", "_")
        .replace(" ", "_")
    )
