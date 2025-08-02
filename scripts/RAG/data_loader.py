import json
import os
import re
from collections import defaultdict

import jieba
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 取得當前檔案的絕對路徑
CUR_PATH = os.path.abspath(__file__)
# 你的 data/ 跟 scripts/ 同層，要往上三層
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CUR_PATH)))

# 檔案路徑
tags_path = os.path.join(ROOT_DIR, "data", "embeddings", "tags.json")
embed_path = os.path.join(ROOT_DIR, "data", "embeddings", "embeddings.npy")
index_path = os.path.join(ROOT_DIR, "data", "embeddings", "index.json")
classify_path = os.path.join(ROOT_DIR, "data", "embeddings", "Meat and Vegetarian.json")
CLEAN_ROOT = os.path.join(ROOT_DIR, "data", "clean")


def build_ingredient_set(df_preview: pd.DataFrame, df_detailed: pd.DataFrame) -> set:
    """從 preview_tag 與 ingredient_name 兩欄組成去重後的食材集合"""
    tags_set = set()
    for line in df_preview["preview_tag"]:
        tags_set.update(t.strip() for t in str(line).split(",") if t.strip())
    tags_set.update(df_detailed["ingredient_name"].astype(str).str.strip())
    return {t for t in tags_set if t and not re.fullmatch(r"\d+", t)}


def load_data():
    # 載入 tags, embeddings, index, classify map
    with open(tags_path, "r", encoding="utf-8") as f:
        tags = json.load(f)
    with open(index_path, "r", encoding="utf-8") as f:
        index_map = json.load(f)
    with open(classify_path, "r", encoding="utf-8") as f:
        classify_map = json.load(f)

        # 載入 embedding 相關
    embeddings = np.load(embed_path)
    model = SentenceTransformer("BAAI/bge-m3")
    emb_norms = np.linalg.norm(embeddings, axis=1)

    id2tags = defaultdict(set)
    for item in tags:
        rid = int(item["id"])
        id2tags[rid].add(item["tag"])

    # 動態掃描所有蔬菜資料夾
    vege_names = [
        name
        for name in os.listdir(CLEAN_ROOT)
        if os.path.isdir(os.path.join(CLEAN_ROOT, name))
    ]
    df_cleaned_list, df_preview_list, df_detailed_list, df_steps_list = [], [], [], []
    for v in vege_names:
        folder = os.path.join(CLEAN_ROOT, v)
        df_cleaned_list.append(
            pd.read_csv(
                os.path.join(folder, f"{v}_recipes_cleaned.csv"),
                sep=";",
                encoding="utf-8-sig",
            )
            .rename(columns=lambda x: x.strip())
            .assign(vege_name=v)
        )
        df_preview_list.append(
            pd.read_csv(
                os.path.join(folder, f"{v}_preview_ingredients.csv"),
                encoding="utf-8-sig",
            )
            .rename(columns=lambda x: x.strip())
            .assign(vege_name=v)
        )
        df_detailed_list.append(
            pd.read_csv(
                os.path.join(folder, f"{v}_detailed_ingredients.csv"),
                encoding="utf-8-sig",
            )
            .rename(columns=lambda x: x.strip())
            .assign(vege_name=v)
        )
        df_steps_list.append(
            pd.read_csv(
                os.path.join(folder, f"{v}_recipe_steps.csv"), encoding="utf-8-sig"
            )
            .rename(columns=lambda x: x.strip())
            .assign(vege_name=v)
        )
    # 合併
    df_cleaned = pd.concat(df_cleaned_list, ignore_index=True)
    df_preview = pd.concat(df_preview_list, ignore_index=True)
    df_detailed = pd.concat(df_detailed_list, ignore_index=True)
    df_steps = pd.concat(df_steps_list, ignore_index=True)

    # 建立食材字典 & jieba 自訂字典
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


# 測試專用
if __name__ == "__main__":
    data = load_data()
    print("所有資料已載入！")
    print(f"主表共 {len(data['df_cleaned'])} 筆")
    print(f"預覽表共 {len(data['df_preview'])} 筆")
    print(f"詳細表共 {len(data['df_detailed'])} 筆")
    print(f"步驟表共 {len(data['df_steps'])} 筆")
