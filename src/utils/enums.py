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


class ZeroShotPromptType(Enum):
    V1 = "zs_v1"
    V2 = "zs_v2"
    V3 = "zs_v3"

    def __str__(self):
        return self.value


class RagQPromptType(Enum):
    V1 = "rq_v1"
    V2 = "rq_v2"
    V3 = "rq_v3"
    V4 = "rq_v4"

    def __str__(self):
        return self.value
