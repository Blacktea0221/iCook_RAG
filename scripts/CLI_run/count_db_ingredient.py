import os
import psycopg2

# 從 .env 檔案中讀取資料庫連線資訊 (如果有的話)
# 這裡我直接使用您提供的資訊
DB_HOST = os.environ.get("DB_HOST", "111.184.52.4")
DB_PORT = os.environ.get("DB_PORT", 9527)
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "lorraine")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "0000")

# SQL 查詢語句，用於計算不重複的 id 數量
sql_query = "SELECT COUNT(DISTINCT id) FROM main_recipe;"

# 建立資料庫連線
try:
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    print("資料庫連線成功！")

    # 建立一個游標物件來執行 SQL 語句
    cur = conn.cursor()

    # 執行 SQL 查詢
    cur.execute(sql_query)

    # 取得查詢結果
    unique_id_count = cur.fetchone()[0]

    print(f"在 'ingredient' 資料表中，不重複的 id 數量為：{unique_id_count}")

except (Exception, psycopg2.Error) as error:
    print("資料庫連線失敗或執行錯誤：", error)

finally:
    # 關閉游標和連線
    if "conn" in locals() and conn:
        cur.close()
        conn.close()
        print("資料庫連線已關閉。")
