from enum import Enum


class VQAStrategyType(Enum):
    ZERO_SHOT = "Zero-Shot"
    RAG_Q = "RAG Q"
    RAG_Q_AS = "RAG Q+As"
    RAG_IMG = "RAG IMG"
    RAG_DB_RERANKER = "RAG DB-Reranker"
