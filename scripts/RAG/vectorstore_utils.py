# scripts/RAG/vectorstore_utils.py
import os

import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector

# LangChain 0.3 之後：HuggingFaceEmbeddings 在 langchain_huggingface 套件
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---- 環境變數 ----
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_URL")
TABLE_NAME = os.getenv("PGVECTOR_TABLE", "recipe_vectors")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ---- 全域 Embeddings ----
EMBEDDINGS = HuggingFaceEmbeddings(model_name=MODEL_NAME)


def get_vectorstore() -> PGVector:
    """建立並回傳 PGVector VectorStore 實例（不會刪表）"""
    return PGVector(
        connection_string=PGVECTOR_CONNECTION_STRING,
        embedding_function=EMBEDDINGS,
        collection_name=TABLE_NAME,
        # 建議先不要帶不相容參數，確保版本相容
        # pre_delete_collection=False,
    )


def embed_text_to_np(text: str) -> np.ndarray:
    """用和資料庫相同的 Embedding 模型把輸入句子轉成 1D numpy array(float32)"""
    vec = EMBEDDINGS.embed_query(text)  # list[float]
    return np.asarray(vec, dtype=np.float32)
