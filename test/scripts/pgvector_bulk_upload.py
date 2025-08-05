import json
import os

import numpy as np
import psycopg2
import psycopg2.extras
from tqdm import tqdm

# === 輸入檔案路徑 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 專案根目錄
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")

TAGS_PATH = os.path.join(EMBEDDINGS_DIR, "tags.json")
EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")

# === PostgreSQL 連線設定 ===
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "postgres"
PG_USER = "postgres"
PG_PASSWORD = "0000"
TABLE_NAME = "recipe_vectors"

# === 1. 載入資料 ===
with open(TAGS_PATH, "r", encoding="utf-8") as f:
    tags = json.load(f)
embeddings = np.load(EMBED_PATH)
assert len(tags) == len(embeddings), "資料筆數不一致！"

# === 2. 建立連線 ===
conn = psycopg2.connect(
    host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
)
cur = conn.cursor()

# === 3. 清空資料表（可選，避免重複）===
cur.execute(f"TRUNCATE {TABLE_NAME};")
conn.commit()

# === 4. 批次插入 ===
insert_sql = (
    f"INSERT INTO {TABLE_NAME} (recipe_id, tag, vege_name, embedding) "
    f"VALUES (%s, %s, %s, %s)"
)

batch_size = 1000
batch = []
for idx, (tag_item, emb) in enumerate(tqdm(zip(tags, embeddings), total=len(tags))):
    # psycopg2 需將 numpy array 轉成 list
    emb_list = emb.tolist()
    batch.append(
        (
            tag_item["id"],  # recipe_id
            tag_item["tag"],  # tag
            tag_item["vege_name"],  # vege_name
            emb_list,  # embedding (會自動轉 pgvector)
        )
    )

    # 每 batch_size 筆 bulk insert
    if len(batch) == batch_size or idx == len(tags) - 1:
        psycopg2.extras.execute_batch(cur, insert_sql, batch)
        conn.commit()
        batch = []

print("✅ Bulk insert 完成！")

cur.close()
conn.close()
