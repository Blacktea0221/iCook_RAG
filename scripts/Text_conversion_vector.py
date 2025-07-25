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

# 設定蔬菜名稱 
category = "高麗菜"   # 要處理哪個蔬菜資料夾？（高麗菜就填"高麗菜"）

# 設定輸入與輸出路徑
INPUT_CSV = f"data/clean/{category}/{category}_preview_ingredients.csv"  # 原始 CSV 檔案，需包含 id 與 preview_tag 欄
OUTPUT_DIR = f"data/embeddings/{category}"  # 儲存向量與對應 id、標籤的資料夾
TAGS_JSON = "tags.json"  # 儲存 id 與標籤對應的 JSON
EMBEDDINGS_NPY = "embeddings.npy"  # 儲存 embedding 陣列的 NumPy 檔案
MODEL_NAME = "BAAI/bge-m3"  # 向量化模型，可依需求替換


def main():
    # 1. 讀取 CSV，保留 id 與 preview_tag 並去重
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df = df[["id", "preview_tag"]].drop_duplicates()

    # 2. 擷取標籤與對應 id 的列表
    ids = df["id"].tolist()
    tags = df["preview_tag"].tolist()

    # 3. 初始化向量化模型並產生 embeddings
    print(f"加載模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"開始向量化 {len(tags)} 個標籤...")
    embeddings = model.encode(tags, show_progress_bar=True)

    # 4. 建立輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. 儲存 id 與標籤對應清單 (JSON)
    entries = [{"id": id_, "tag": tag_} for id_, tag_ in zip(ids, tags)]
    with open(os.path.join(OUTPUT_DIR, TAGS_JSON), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"已儲存標籤對應檔: {os.path.join(OUTPUT_DIR, TAGS_JSON)}")

    # 6. 儲存 embeddings 陣列 (NumPy)
    np.save(os.path.join(OUTPUT_DIR, EMBEDDINGS_NPY), embeddings)
    print(f"已儲存 embeddings 檔: {os.path.join(OUTPUT_DIR, EMBEDDINGS_NPY)}")

    print("所有向量化流程完成！")


if __name__ == "__main__":
    main()
