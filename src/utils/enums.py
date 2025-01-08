from enum import Enum


class VQAStrategyType(Enum):
    ZERO_SHOT = "zero_shot"
    RAG_Q = "rag_q"
    RAG_Q_AS = "rag_q_as"
    RAG_IMG = "rag_img"
    RAG_DB_RERANKER = "rag_db_reranker"

    def __str__(self):
        return self.value


class WikiMedRepresentationMode(Enum):
    SENTENCE = 'sentence'
    WORD = 'word'
