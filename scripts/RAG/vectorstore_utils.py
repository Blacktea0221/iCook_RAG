# vectorstore_utils.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import numpy as np

# -- 向量資料庫連線設定 --
PGVECTOR_CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:0000@localhost:5432/postgres"
)
TABLE_NAME = "recipe_vectors"
EMBEDDING_DIM = 1024  # 你的向量維度

# -- Embedding Model 設定 --
MODEL_NAME = "BAAI/bge-m3"  # 你目前用的模型
EMBEDDINGS = CommunityEmbeddings(
    model_name=MODEL_NAME
)  # 如果有用 sentence-transformers，可直接使用


def get_vectorstore():
    """
    建立並回傳 PGVector VectorStore 實例
    """
    vectorstore = PGVector(
        connection_string=PGVECTOR_CONNECTION_STRING,
        embedding_function=EMBEDDINGS,
        collection_name=TABLE_NAME,
        pre_delete_collection=False,
    )
    return vectorstore


# def search_vectorstore(query, top_k=5):
#     """
#     查詢相似的食譜tag
#     """
#     vectorstore = get_vectorstore()
#     # 查詢會回傳一組 Document 物件
#     results = vectorstore.similarity_search(query, k=top_k)
#     # 結果格式轉換（根據你需要的欄位）
#     output = []
#     for doc in results:
#         meta = doc.metadata
#         # metadata 通常含有id, tag, vege_name等
#         output.append(
#             {
#                 "id": meta.get("recipe_id"),
#                 "tag": meta.get("tag"),
#                 "vege_name": meta.get("vege_name"),
#                 "score": getattr(doc, "score", None),
#                 "text": getattr(doc, "page_content", None),
#             }
#         )
#     return output

def embed_text_to_np(text: str) -> np.ndarray:
    """
    用和資料庫相同的 Embedding 模型（bge-m3）把輸入句子轉成 1D numpy array
    """
    vec = EMBEDDINGS.embed_query(text)  # list[float]
    return np.asarray(vec, dtype="float32")
