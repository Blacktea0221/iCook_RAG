import psycopg2

# 替換為你的密碼
PASSWORD = "0000"

try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="lorraine",
        password=PASSWORD
    )
    print("✅ 成功連線到 PostgreSQL 資料庫")

    # 建立 cursor 並查詢一張表（例如 public.recipe_steps）
    cur = conn.cursor()
    cur.execute("SELECT * FROM public.recipe_steps LIMIT 5;")
    rows = cur.fetchall()

    for row in rows:
        print(row)

    cur.close()
    conn.close()

except Exception as e:
    print("❌ 發生錯誤：", e)
