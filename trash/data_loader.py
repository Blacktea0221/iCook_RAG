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
    df_cleaned = fetch_df_from_db("SELECT id, recipe, vege_id FROM public.main_recipe;")
    df_preview = fetch_df_from_db("SELECT recipe_id, ingredient, preview_tag FROM public.ingredient;")
    df_detailed = fetch_df_from_db("SELECT recipe_id, step_no, description FROM public.recipe_steps;")

    # 【可選】如果之後要顯示蔬菜名，可同時載 basic_vege
    # df_basic_vege = fetch_df_from_db("SELECT id, vege_name FROM public.basic_vege;")

    ing_set = build_ingredient_set(df_preview, df_detailed)  # 用 ingredient 表就夠
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
        "df_steps": df_detailed,  # ← 如果你 elsewhere 真的需要 df_steps，請改成單獨變數
        "ingredient_set": ing_set,
        "id2tags": _build_id2tags(tags),
        "model": model,
        "embeddings": embeddings,
        "emb_norms": emb_norms,
        # "df_basic_vege": df_basic_vege,  # 若有需要再加
    }


if __name__ == "__main__":
    data = load_data()
    print("所有資料已從資料庫載入！")
    print(f"主表共 {len(data['df_cleaned'])} 筆")
    print(f"預覽表共 {len(data['df_preview'])} 筆")
    print(f"詳細表共 {len(data['df_detailed'])} 筆")
    print(f"步驟表共 {len(data['df_steps'])} 筆")
