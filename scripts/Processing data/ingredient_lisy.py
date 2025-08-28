import os
import json
import psycopg2
import pandas as pd
from dotenv import load_dotenv

# --- 載入環境變數 ---
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# --- 連線資料庫 ---
conn = psycopg2.connect(**DB_CONFIG)

# --- SQL 抓取資料 ---
query = "SELECT recipe_id, preview_tag FROM ingredient;"
df = pd.read_sql(query, conn)

# --- 聚合成 dict ---
result_dict = df.groupby("recipe_id")["preview_tag"].apply(list).to_dict()

# --- 設定輸出路徑 ---
out_dir = "data/embeddings"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "ingredient_list.json")

# --- 存成 JSON ---
with open(out_path, "w", encoding="utf-8") as jf:
    json.dump(result_dict, jf, ensure_ascii=False, indent=2)

print(f"JSON 生成位置：{os.path.abspath(out_path)}")

# --- 關閉連線 ---
conn.close()
