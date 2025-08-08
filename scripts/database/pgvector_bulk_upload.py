import json
import os

import numpy as np
import psycopg2
import psycopg2.extras
from tqdm import tqdm

# 修改成你的資料夾路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
TAGS_PATH = os.path.join(EMBEDDINGS_DIR, "tags.json")
EMBED_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")

# PostgreSQL 連線設定
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "postgres"
PG_USER = "lorraine"
PG_PASSWORD = "0000"
TABLE_NAME = "ingredient_vectors"  # 假設你用的表名

# 載入資料
with open(TAGS_PATH, "r", encoding="utf-8") as f:
    tags = json.load(f)
embeddings = np.load(EMBED_PATH)
assert len(tags) == len(embeddings), "資料筆數不一致！"

# 建立連線
conn = psycopg2.connect(
    host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
)
cur = conn.cursor()

# 清空資料表（可選）
cur.execute(f"TRUNCATE {TABLE_NAME};")
conn.commit()

insert_sql = (
    f"INSERT INTO {TABLE_NAME} (recipe_id, tag, vege_name, embedding) "
    f"VALUES (%s, %s, %s, %s) "
    f"ON CONFLICT (recipe_id, tag) DO UPDATE SET "
    f"vege_name = EXCLUDED.vege_name, "
    f"embedding = EXCLUDED.embedding"
)

batch_size = 1000
batch = []
for idx, (tag_item, emb) in enumerate(tqdm(zip(tags, embeddings), total=len(tags))):
    emb_list = emb.tolist()
    batch.append(
        (
            tag_item["id"],
            tag_item["tag"],
            tag_item["vege_name"],
            emb_list,
        )
    )
    if len(batch) == batch_size or idx == len(tags) - 1:
        psycopg2.extras.execute_batch(cur, insert_sql, batch)
        conn.commit()
        batch = []

print("✅ Bulk insert 完成！")

cur.close()
conn.close()
