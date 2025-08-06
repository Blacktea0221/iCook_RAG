import json
import os
import re
from collections import defaultdict

import jieba
import numpy as np
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer

# 取得當前檔案的絕對路徑
CUR_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CUR_PATH)))

# 檔案路徑
tags_path = os.path.join(ROOT_DIR, "data", "embeddings", "tags.json")
embed_path = os.path.join(ROOT_DIR, "data", "embeddings", "embeddings.npy")
index_path = os.path.join(ROOT_DIR, "data", "embeddings", "index.json")
classify_path = os.path.join(ROOT_DIR, "data", "embeddings", "Meat and Vegetarian.json")

# 資料庫連線設定
DB_CONFIG = dict(
    host="localhost", port=5432, database="postgres", user="lorraine", password="0000"
)


def fetch_df_from_db(query: str) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def build_ingredient_set(df_preview: pd.DataFrame, df_detailed: pd.DataFrame) -> set:
    tags_set = set()
    for line in df_preview["preview_tag"]:
        tags_set.update(t.strip() for t in str(line).split(",") if t.strip())
    tags_set.update(df_detailed["ingredient_name"].astype(str).str.strip())
    return {t for t in tags_set if t and not re.fullmatch(r"\d+", t)}


def load_data():
    with open(tags_path, "r", encoding="utf-8") as f:
        tags = json.load(f)
    with open(index_path, "r", encoding="utf-8") as f:
        index_map = json.load(f)
    with open(classify_path, "r", encoding="utf-8") as f:
        classify_map = json.load(f)

    embeddings = np.load(embed_path)
    model = SentenceTransformer("BAAI/bge-m3")
    emb_norms = np.linalg.norm(embeddings, axis=1)

    id2tags = defaultdict(set)
    for item in tags:
        rid = int(item["id"])
        id2tags[rid].add(item["tag"])

    df_cleaned = fetch_df_from_db("SELECT * FROM public.recipes_cleaned;")
    df_preview = fetch_df_from_db("SELECT * FROM public.preview_ingredients;")
    df_detailed = fetch_df_from_db("SELECT * FROM public.detailed_ingredients;")
    df_steps = fetch_df_from_db("SELECT * FROM public.recipe_steps;")

    ing_set = build_ingredient_set(df_preview, df_detailed)
    for w in ing_set:
        jieba.add_word(w)
    print(f"[data_loader] 食材字典已初始化（共 {len(ing_set)} 項）")

    return {
        "tags": tags,
        "index_map": index_map,
        "classify_map": classify_map,
        "df_cleaned": df_cleaned,
        "df_preview": df_preview,
        "df_detailed": df_detailed,
        "df_steps": df_steps,
        "ingredient_set": ing_set,
        "id2tags": id2tags,
        "model": model,
        "embeddings": embeddings,
        "emb_norms": emb_norms,
    }


if __name__ == "__main__":
    data = load_data()
    print("所有資料已從資料庫載入！")
    print(f"主表共 {len(data['df_cleaned'])} 筆")
    print(f"預覽表共 {len(data['df_preview'])} 筆")
    print(f"詳細表共 {len(data['df_detailed'])} 筆")
    print(f"步驟表共 {len(data['df_steps'])} 筆")
