import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer

# PostgreSQL 連線參數（請自行調整）

PG_HOST = os.getenv("DB_HOST")
PG_PORT = int(os.getenv("DB_PORT", "5432"))
PG_DB = os.getenv("DB_NAME")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASSWORD")

OUTPUT_DIR = "data/embeddings"
TAGS_JSON = "tags.json"
EMBEDDINGS_NPY = "embeddings.npy"
INDEX_JSON = "index.json"
MODEL_NAME = "BAAI/bge-m3"


def fetch_data():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    )
    query = """
    SELECT i.recipe_id, i.preview_tag, m.vege_id
    FROM ingredient i
    LEFT JOIN main_recipe m ON i.recipe_id = m.id
    WHERE i.preview_tag IS NOT NULL AND i.preview_tag <> ''
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def main():
    print("從資料庫抓取資料...")
    df = fetch_data()
    df = df.drop_duplicates(subset=["recipe_id", "preview_tag"])

    all_ids = df["recipe_id"].tolist()
    all_tags = df["preview_tag"].tolist()
    # 如果 vege_id 有空，改成 unknown 字串
    all_veges = df["vege_id"].fillna("unknown").astype(str).tolist()

    print(f"共 {len(all_tags)} 筆標籤需要向量化")

    print(f"加載模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(all_tags, show_progress_bar=True)

    # 產生 index.json 依 vege_id 分段 (group by vege_id)
    index_map = {}
    start = 0
    vege_group = defaultdict(list)
    for v in all_veges:
        vege_group[v].append(1)
    # 重新計算每個 vege_id 的起始位置和長度
    index_map = {}
    start = 0
    for vege_id, items in vege_group.items():
        length = len(items)
        index_map[vege_id] = {"start": start, "length": length}
        start += length

    # 儲存輸出
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entries = [
        {"id": _id, "tag": _tag, "vege_name": _v}
        for _id, _tag, _v in zip(all_ids, all_tags, all_veges)
    ]

    with open(os.path.join(OUTPUT_DIR, TAGS_JSON), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"已儲存 {TAGS_JSON}")

    np.save(os.path.join(OUTPUT_DIR, EMBEDDINGS_NPY), embeddings)
    print(f"已儲存 {EMBEDDINGS_NPY}")

    with open(os.path.join(OUTPUT_DIR, INDEX_JSON), "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)
    print(f"已儲存 {INDEX_JSON}")

    print("向量化與檔案輸出完成！")


if __name__ == "__main__":
    main()
