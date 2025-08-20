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

# --- 可自行維護葷食相關清單 ---
MEAT_INGREDIENTS = [
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
    "雞肉",
    "雞腿",
    "雞翅",
    "牛肉",
    "牛排",
    "羊肉",
    "魚",
    "蝦",
    "螃蟹",
    "蛤蜊",
    "花枝",
    "干貝",
    "小魚乾",
    "魚板",
    "魚漿",
    "章魚",
    "肉燥",
    "肉鬆",
    "雞蛋",
    "雞胸肉",
    "雞塊",
    "鴨肉",
    "鵝肉",
    "豬舌",
    "豬肚",
    "豬血",
    "雞胗",
    "鴨血",
    "鴨腸",
    "雞心",
    "鵝肝",
    "鮑魚",
    "海參",
    "魚翅",
    "魚肚",
    "燕窩",
    "魷魚",
    "鮮蝦",
    "蚵仔",
    "生蠔",
    "淡菜",
    "鯖魚",
    "鮭魚",
    "鮪魚",
    "鯛魚",
    "秋刀魚",
    "虱目魚",
    "土雞肉",
    "烏骨雞",
    "鵝肉",
    "鴨腸",
    "牛筋",
    "牛肚",
    "牛腱",
    "羊肉串",
    "羊排",
    "羊腩",
    "兔肉",
    "鹿肉",
    "豬油渣",
    "蝦米",
    "蝦皮",
    "魚卵",
    "蟹肉",
    "蟹膏",
    "飛魚卵",
    "烏魚子",
    "干貝醬",
    "魚露",
    "蝦醬",
    "沙茶醬",
    "雞精",
    "雞粉",
    "雞湯塊",
    "豬油",
    "牛油",
    "肉骨湯",
    "豬肉片",
    "牛肉片",
    "羊肉片",
    "雞肉塊",
    "牛舌",
    "牛尾",
    "豬頭皮",
    "豬大腸",
    "豬小腸",
    "豬肺",
    "雞冠",
    "雞皮",
    "鴨掌",
    "鴨舌",
    "鯊魚",
    "龍蝦",
    "田雞",
    "海蜇皮",
    "海膽",
    "扇貝",
    "螺肉",
    "鮪魚罐頭",
    "鯖魚罐頭",
    "魚鬆",
    "雞肉丸",
    "豬肉丸",
    "魚丸",
    "蝦丸",
    "花枝丸",
    "獅子頭",
    "德國香腸",
    "熱狗",
    "火腿片",
    "培根片",
    "雞柳",
    "豬絞肉",
    "牛肉燥",
    "豬肉條",
    "雞脖子",
    "鴨脖子",
    "鵝掌",
    "鵝頭",
    "豬腳筋",
    "牛骨",
    "雞骨",
    "魚骨",
    "肉骨",
    "雞湯",
    "魚高湯",
    "蝦高湯",
    "肉高湯",
    "牛高湯",
    "豬骨湯",
    "老母雞湯",
    "火雞肉",
    "鴿肉",
    "鵪鶉蛋",
    "魚餃",
    "蝦餃",
    "蛋餃",
    "燕餃",
    "貢丸湯",
    "排骨酥",
    "肉羹",
    "紅燒肉",
    "滷肉",
    "三層肉",
    "蝦皮",
    "蝦米",
]

PORK_INGREDIENTS = [
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
    "豬舌",
    "豬肚",
    "豬血",
    "豬油渣",
    "豬肉片",
    "豬頭皮",
    "豬大腸",
    "豬小腸",
    "豬肺",
    "豬絞肉",
    "豬肉條",
    "豬骨",
    "肉骨",
    "豬骨湯",
    "豬肉丸",
    "獅子頭",
    "德國香腸",
    "熱狗",
    "火腿片",
    "培根片",
    "豬腳筋",
    "肉骨茶湯料",
    "豬油",
    "豬骨湯塊",
    "豬骨高湯",
    "豬骨粉",
    "豬肉粉",
    "豬肉精粉",
    "豬肉湯塊",
    "豬油渣粉",
    "豬骨濃縮液",
    "豬骨精",
]

NON_VEGE_SEASONINGS = [
    "魚露",
    "魚醬",
    "魚湯塊",
    "魚精粉",
    "魚粉",
    "魚高湯",
    "魚膠原蛋白粉",
    "沙丁魚醬",
    "鯷魚醬",
    "鯷魚油",
    "蝦醬",
    "蝦米粉",
    "蝦仁粉",
    "蝦湯塊",
    "蝦膏",
    "蝦油",
    "蟹黃醬",
    "蟹膏",
    "蟹油",
    "蟹精粉",
    "雞精",
    "雞粉",
    "雞湯塊",
    "雞湯粉",
    "雞油",
    "雞骨高湯",
    "雞高湯粉",
    "雞濃縮液",
    "雞肉醬",
    "雞汁",
    "豬油",
    "豬骨湯塊",
    "豬骨高湯",
    "豬骨粉",
    "豬肉粉",
    "豬肉精粉",
    "豬肉湯塊",
    "豬油渣粉",
    "豬骨濃縮液",
    "豬骨精",
    "牛油",
    "牛骨高湯",
    "牛骨粉",
    "牛肉粉",
    "牛肉湯塊",
    "牛精粉",
    "牛肉濃縮液",
    "牛骨濃縮液",
    "牛油膏",
    "牛肉醬",
    "鴨油",
    "鴨湯塊",
    "鴨粉",
    "鴨骨高湯",
    "鴨骨精粉",
    "鴨肉濃縮液",
    "鵝油",
    "鵝肝醬",
    "鵝高湯粉",
    "鵝肉精",
    "羊油",
    "羊骨高湯",
    "羊肉粉",
    "羊肉湯塊",
    "羊精粉",
    "羊骨粉",
    "羊肉濃縮液",
    "羊骨濃縮液",
    "羊脂膏",
    "羊骨精",
    "海鮮醬",
    "海鮮粉",
    "海鮮湯塊",
    "海鮮高湯",
    "海鮮精粉",
    "海鮮濃縮液",
    "魷魚粉",
    "章魚粉",
    "干貝粉",
    "干貝醬",
    "肉骨茶湯料",
    "沙茶醬",
    "XO醬",
    "蠔油",
    "鰻魚醬",
    "鰹魚粉",
    "鰹魚高湯",
    "柴魚片",
    "柴魚粉",
    "柴魚濃縮液",
    "牛排醬",
    "烤肉醬",
    "羊肉爐湯底",
    "牛肉爐湯底",
    "鴨肉爐湯底",
    "豬肉爐湯底",
    "麻辣鍋底",
    "高湯包",
    "日式拉麵湯底",
    "火鍋湯精",
]

FIVE_SPICE = [
    "蔥",
    "青蔥",
    "大蔥",
    "小蔥",
    "細香蔥",
    "紅蔥頭",
    "洋蔥",
    "黃洋蔥",
    "白洋蔥",
    "紫洋蔥",
    "乾蔥",
    "蔥白",
    "蔥花",
    "蔥段",
    "蔥油",
    "蔥末",
    "蔥頭",
    "蔥葉",
    "蔥苗",
    "小洋蔥",
    "甜洋蔥",
    "珍珠洋蔥",
    "扁蔥",
    "日本蔥",
    "韓國大蔥",
    "蒜",
    "大蒜",
    "蒜頭",
    "蒜瓣",
    "蒜泥",
    "蒜末",
    "蒜苗",
    "蒜苔",
    "蒜黃",
    "黑蒜",
    "蒜油",
    "蒜粉",
    "蒜蓉",
    "蒜片",
    "蒜香醬",
    "獨子蒜",
    "象拔蒜",
    "紫皮蒜",
    "乾蒜",
    "鹽漬蒜",
    "蒜醬",
    "韭菜",
    "韭菜花",
    "韭菜苔",
    "韭黃",
    "韭芽",
    "野韭",
    "韭菜籽",
    "韭花醬",
    "韭菜根",
    "冬韭",
    "夏韭",
    "山韭",
    "韭白",
    "韭齡",
    "紫韭",
    "興渠",
    "薤",
    "薤白",
    "薤葉",
    "野蔥",
    "洋韭",
    "慈蔥",
    "蘭蔥",
    "岩蔥",
    "鹿蔥",
    "薤根",
    "薤頭",
    "假蔥",
    "野蒜",
    "山蒜",
    "胡蔥",
    "球莖蔥",
    "蛇蔥",
    "高山韭",
    "野薤",
]

# --- 系統提示 ---
system_prompt_simple = (
    "你是一個嚴格的菜譜分類機器人，唯一的任務是僅根據提供的規則，將食譜分類為『素食』或『葷食』。\n"
    "\n"
    "分類規則：\n"
    "- 葷食：食材或調味料中，只要符合任一以下條件，即判斷為葷食。\n"
    "  - 任何肉類、魚類或海鮮成分。\n"
    "  - 五辛：包含蔥、蒜、韭菜、洋蔥、香菜。\n"
    "  - 特定調味料：魚露、蝦醬、沙茶醬、雞精、雞粉、雞湯塊、豬油、肉骨湯。\n"
    "- 素食：不符合上述所有葷食條件的食譜，即判斷為素食。\n"
    "\n"
    "請嚴格遵守以下輸出格式：\n"
    "- 格式：每行僅包含一個食譜ID、一個空格、和最終分類結果（素食/葷食）。\n"
    "- 範例：123 葷食\n"
    "- 重要提示：除了 ID 和分類，不允許任何其他文字、符號、標點、解釋或額外的換行。"
)


# --- 從資料庫抓取食材 ---
def fetch_ingredients():
    conn = psycopg2.connect(**DB_CONFIG)
    # === 測試模式抓前 200 筆 ===
    query = "SELECT recipe_id, preview_tag FROM ingredient LIMIT 50 OFFSET 1500;"
    # === 正式模式抓全部 ===
    # query = "SELECT recipe_id, preview_tag FROM ingredient;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# --- 批次分類函數 ---
def classify_batch(batch_df: pd.DataFrame) -> dict:
    user_prompt = ""
    for row in batch_df.itertuples():
        user_prompt += f"{row.recipe_id} {row.preview_tag}\n"

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
        match = re.match(r"(\S+)\s*(素食|葷食)", line)
        if match:
            rid, label = match.groups()
            ingredients = batch_df.loc[
                batch_df["recipe_id"] == rid, "preview_tag"
            ].values[0]

            # 從 LLM 判斷葷素
            vegetarian = True if label == "素食" else False

            # 額外判斷是否含豬肉，使用 PORK_INGREDIENTS 清單
            uses_pork = any(re.search(prod, ingredients) for prod in PORK_INGREDIENTS)

            # 您原有的這段判斷邏輯可以移除，因為 LLM 已經負責葷素分類
            # if any(re.search(item, ingredients) for item in NON_VEGE_SEASONINGS):
            #     vegetarian = False
            # 這裡的 'NON_VEGE_SEASONINGS' 應該已經涵蓋在 LLM 的判斷規則中

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

out_path = os.path.join(out_dir, "Meat_and_Vegetarian_1500.json")
with open(out_path, "w", encoding="utf-8") as jf:
    json.dump(result_dict, jf, ensure_ascii=False, indent=2)

print(f"JSON 生成位置：{os.path.abspath(out_path)}")
