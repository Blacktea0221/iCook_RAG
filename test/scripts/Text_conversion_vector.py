#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vectorize_preview_tags.py

此腳本用於：
1. 讀取 "九層塔_preview_ingredients.csv" 中的預覽食材標籤 (preview_tag) 及其對應食譜 id
2. 使用 SentenceTransformer 模型將每個標籤轉換為向量 (embeddings)
3. 將結果儲存至指定資料夾 "preview_embeddings" 中，並保留原始 id

使用前請安裝：
$ pip install pandas numpy sentence-transformers

執行：
$ python vectorize_preview_tags.py
"""

import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 掃描所有已清理的蔬菜資料夾
INPUT_ROOT = "data/clean"  # 根目錄下每個子資料夾皆為一種 vege_name
OUTPUT_DIR = (
    "data/embeddings"  # 最終只會在這裡放三個檔案：tags.json, embeddings.npy, index.json
)
TAGS_JSON = "tags.json"  # 儲存 id 與標籤對應的 JSON
EMBEDDINGS_NPY = "embeddings.npy"  # 儲存 embedding 陣列的 NumPy 檔案
INDEX_JSON = "index.json"
MODEL_NAME = "BAAI/bge-m3"  # 向量化模型，可依需求替換


def main():
    # 讀取單一檔案 ingredient.csv
    csv_path = "data/clean/ingredient.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")[
        ["recipe_id", "preview_tag"]
    ].drop_duplicates()

    all_ids = df["recipe_id"].tolist()
    all_tags = df["preview_tag"].tolist()

    # 所有向量屬於同一類別 'ingredient'
    all_veges = ["ingredient"] * len(all_tags)
    index_map = {"ingredient": {"start": 0, "length": len(all_tags)}}

    # 向量化
    print(f"加載模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"開始向量化 {len(all_tags)} 個標籤...")
    embeddings = model.encode(all_tags, show_progress_bar=True)

    # 儲存輸出
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entries = [
        {"id": _id, "tag": _tag, "vege_name": _v}
        for _id, _tag, _v in zip(all_ids, all_tags, all_veges)
    ]
    with open(os.path.join(OUTPUT_DIR, TAGS_JSON), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"已儲存標籤對應檔: {os.path.join(OUTPUT_DIR, TAGS_JSON)}")

    np.save(os.path.join(OUTPUT_DIR, EMBEDDINGS_NPY), embeddings)
    print(f"已儲存 embeddings 檔: {os.path.join(OUTPUT_DIR, EMBEDDINGS_NPY)}")

    with open(os.path.join(OUTPUT_DIR, INDEX_JSON), "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)
    print(f"已儲存索引檔: {os.path.join(OUTPUT_DIR, INDEX_JSON)}")

    print("所有向量化流程完成！")


if __name__ == "__main__":
    main()
