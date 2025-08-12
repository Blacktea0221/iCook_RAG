# connect_database.py
import os
import psycopg2
from dotenv import load_dotenv

# 載入 .env（預設讀取與此檔同目錄或專案根目錄的 .env）
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", ""))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def main():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,   # 或 database=DB_NAME 皆可
            user=DB_USER,
            password=DB_PASSWORD,
        )
        print("✅ 成功連線到 PostgreSQL")

        cur = conn.cursor()
        # 你原本就有在測試抓表，可保留；若表不存在可改成下行簡單測試：
        # cur.execute("SELECT version();")
        cur.execute("SELECT * FROM public.recipe_steps LIMIT 5;")
        rows = cur.fetchall()
        for row in rows:
            print(row)

        cur.close()
        conn.close()

    except Exception as e:
        print("❌ 發生錯誤：", e)

if __name__ == "__main__":
    main()
