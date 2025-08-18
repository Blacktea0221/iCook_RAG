import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import re
import openai
from tqdm import tqdm
import json
import math

# --- 讀取環境變數 ---
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

openai.api_key = os.getenv("OPENAI_API_KEY")

# --- 可自行維護豬肉製品清單 ---
PORK_PRODUCTS = [
    "豬肉",
    "豬排",
    "五花肉",
    "絞肉",
    "排骨",
    "培根",
    "火腿",
    "香腸",
    "臘肉",
    "貢丸",
    "豬肋排",
    "豬里脊",
    "豬腿肉",
    "豬腰",
    "豬肝",
    "豬心",
    "豬腳",
    "豬耳朵",
    "豬尾巴",
    "豬油",
]

# --- 系統提示 ---
system_prompt_simple = (
    "你是一個菜譜分類器，僅能二選一：素食 或 葷食。\n"
    "判斷規則：如任一食材含肉類、魚類、海鮮或五辛（蔥, 蒜, 韭菜, 洋蔥, 香菜），或含下列調味料：魚露, 蝦醬, 沙茶醬, 雞精, 雞粉, 雞湯塊, 豬油, 肉骨湯，則判定為葷食；否則為素食。\n"
    "請先依食譜 ID 聚合該食譜所有食材，再進行判斷。\n"
    "請**只**回傳每筆食譜分類（素食 或 葷食），每筆一行，不要多餘文字、解釋或引號。\n"
    "格式：每行開頭為食譜 ID，再空格，再分類。例如：123 葷食"
)

# --- 非素食調味料 ---
NON_VEGE_SEASONINGS = [
    "魚露",
    "蝦醬",
    "沙茶醬",
    "雞精",
    "雞粉",
    "雞湯塊",
    "豬油",
    "肉骨湯",
]


# --- 從資料庫抓取食材 ---
def fetch_ingredients():
    conn = psycopg2.connect(**DB_CONFIG)
    # === 測試模式抓前 200 筆 ===
    # query = "SELECT recipe_id, preview_tag FROM ingredient LIMIT 51 OFFSET 999;"
    # === 正式模式抓全部 ===
    query = "SELECT recipe_id, preview_tag FROM ingredient;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# --- 批次分類函數 ---
def classify_batch(batch_df: pd.DataFrame) -> dict:
    # 準備 user prompt
    user_prompt = ""
    for row in batch_df.itertuples():
        user_prompt += f"{row.recipe_id} {row.preview_tag}\n"

    # 呼叫模型
    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt_simple},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw_text = response.choices[0].message.content.strip()
    lines = raw_text.splitlines()

    result = {}
    for line in lines:
        # line 範例: "123 葷食"
        match = re.match(r"(\S+)\s*(素食|葷食)", line)
        if match:
            rid, label = match.groups()
            ingredients = batch_df.loc[
                batch_df["recipe_id"] == rid, "preview_tag"
            ].values[0]
            vegetarian = True if label == "素食" else False
            uses_pork = any(re.search(prod, ingredients) for prod in PORK_PRODUCTS)
            # 補強：非素食調味料
            if any(re.search(item, ingredients) for item in NON_VEGE_SEASONINGS):
                vegetarian = False
            result[rid] = {"vegetarian": vegetarian, "uses_pork": uses_pork}
    return result


# --- 執行分類 ---
df = fetch_ingredients()
result_dict = {}

batch_size = 50
num_batches = math.ceil(len(df) / batch_size)

for i in tqdm(range(num_batches), desc="批次分類中"):
    batch_df = df.iloc[i * batch_size : (i + 1) * batch_size]
    batch_result = classify_batch(batch_df)
    result_dict.update(batch_result)

# --- 4. 存 JSON（保留原段落） ---
out_dir = "data/embeddings"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "Meat_and_Vegetarian.json")
with open(out_path, "w", encoding="utf-8") as jf:
    json.dump(result_dict, jf, ensure_ascii=False, indent=2)

print(f"JSON 生成位置：{os.path.abspath(out_path)}")
