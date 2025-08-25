import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import re
import openai
from tqdm import tqdm
import json
import math

# --- 設定 ---
# 設為 True 則在 JSON 中包含 "reason" 欄位，方便除錯；設為 False 則只輸出 vegetarian 和 uses_pork。
ENABLE_REASON_LOGGING = True

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

# --- 豬肉製品清單 (for uses_pork) ---
PORK_PRODUCTS = [
    "豬肉", "豬排", "五花肉", "絞肉", "排骨", "培根", "火腿", "香腸", "臘肉", "貢丸",
    "豬肋排", "豬里脊", "豬腿肉", "豬腰", "豬肝", "豬心", "豬腳", "豬耳朵", "豬尾巴",
    "豬油", "豬肚", "豬腎", "豬腦", "豬舌", "梅花肉", "松阪豬", "豬頸肉", "肉乾",
    "肉鬆", "肉脯", "肉丸", "紅燒肉", "東坡肉", "豬皮", "豬血糕", "豬血", "豬大腸",
    "豬小腸", "大腸頭", "粉腸","豬","豚"
]

# --- 非素食調味料及其他葷食關鍵字清單 ---
NON_VEGE_KEYWORDS = [
    "魚露", "蝦醬", "沙茶醬", "雞精", "雞粉", "雞湯塊", "蠔油", "干貝醬", "烏魚子醬",
    "XO醬", "海鮮醬", "雞油", "大骨湯", "滷肉燥", "肉燥", "牛油", "牛肉湯塊",
    "蔥", "蒜", "洋蔥", "韭菜", "蕎頭", "大蒜粉", "洋蔥粉",
    "雞肉", "牛肉", "羊肉", "魚肉", "海鮮", "蝦仁", "蟹肉", "雞蛋", "肉骨",
    "蝦米", "肉鬆", "肉脯", "吻仔魚", "肉燥", "滷肉"
]

# --- 系統提示 (LLM 葷素判斷) ---
system_prompt_llm_vege = (
    "你是一個嚴格且精準的菜譜分類器，任務是將食譜分類為「素食」或「葷食」。\n"
    "請務必仔細審查**所有**食材與調味料，並根據以下規則進行**嚴格判斷**：\n"
    "1. 如果食譜中包含任何肉類（如牛、羊、豬、雞、鴨等）、魚類、海鮮，則立即判斷為「葷食」。\n"
    "2. 如果食譜中包含魚露, 蝦醬, 蠔油, 沙茶醬, 雞精, 雞粉, 雞湯塊, 豬油, 肉骨湯, 滷肉燥, 大骨湯等非素食調味料或製品，則判斷為「葷食」。\n"
    "3. 如果食譜中包含五辛（蔥、蒜、韭菜、洋蔥、韭菜），也必須判斷為「葷食」。\n"
    "4. 只有在**完全不含**上述任何葷食成分的情況下，才能判斷為「素食」。\n"
    "請先依食譜 ID 聚合該食譜的所有食材，再進行最終判斷。\n"
    "請**只**回傳食譜的最終分類（素食 或 葷食），每筆一行，**不要**包含任何多餘的文字、解釋或引號。\n"
    "格式範例：\n"
    "123 葷食\n"
    "456 素食"
)

# --- 從資料庫抓取食材 ---
def fetch_ingredients() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    # === 測試模式抓前 200 筆 ===
    query = "SELECT recipe_id, preview_tag FROM ingredient LIMIT 51 OFFSET 600;"
    # === 正式模式抓全部 ===
    # query = "SELECT recipe_id, preview_tag FROM ingredient;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- 根據關鍵字判斷葷素及豬肉 ---
def classify_by_keywords(ingredients_list: list[str]) -> dict:
    ingredients_str = " ".join(ingredients_list)
    
    # 判斷是否含豬肉（精準度最高）
    uses_pork = any(re.search(prod, ingredients_str) for prod in PORK_PRODUCTS)
    
    # 判斷是否含任何葷食
    is_non_vege = uses_pork or any(re.search(keyword, ingredients_str) for keyword in NON_VEGE_KEYWORDS)

    if is_non_vege:
        result = {"vegetarian": False, "uses_pork": uses_pork}
        if ENABLE_REASON_LOGGING:
            result["reason"] = "keyword_match"
        return result
    else:
        # 如果關鍵字都沒找到，回傳空字典表示需要LLM判斷
        return {}

# --- LLM 葷素判斷 ---
def classify_with_llm(recipes_df: pd.DataFrame) -> dict:
    if recipes_df.empty: return {}
    user_prompt = ""
    for row in recipes_df.itertuples():
        user_prompt += f"{row.recipe_id} {' '.join(row.preview_tag)}\n"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "system", "content": system_prompt_llm_vege}, {"role": "user", "content": user_prompt}],
            temperature=0,
        )
        lines = response.choices[0].message.content.strip().splitlines()
        result = {}
        for line in lines:
            match = re.match(r"(\S+)\s*(素食|葷食)", line)
            if match:
                rid, label = match.groups()
                result[rid] = True if label == "素食" else False
        return result
    except Exception as e:
        print(f"呼叫 LLM 進行葷素分類發生錯誤: {e}")
        return {}


# --- 主程式 ---
def main():
    print("正在從資料庫讀取食材資料並聚合...")
    df = fetch_ingredients()
    
    recipes = df.groupby('recipe_id')['preview_tag'].apply(list).reset_index()
    
    final_results = {}
    llm_needed_recipes = []
    
    progress_bar = tqdm(recipes.itertuples(), total=len(recipes), desc="正在使用關鍵字進行初篩")

    for row in progress_bar:
        recipe_id = row.recipe_id
        ingredients_list = row.preview_tag
        
        keyword_result = classify_by_keywords(ingredients_list)
        
        if keyword_result:
            result = keyword_result
            if not ENABLE_REASON_LOGGING:
                result.pop("reason", None)
            final_results[recipe_id] = result
        else:
            llm_needed_recipes.append(row)

    # 將需要 LLM 判斷的食譜分批處理
    llm_df = pd.DataFrame(llm_needed_recipes, columns=["Index", "recipe_id", "preview_tag"])
    
    llm_vege_results = classify_with_llm(llm_df)
    
    # 整合所有結果
    for rid in tqdm(llm_df['recipe_id'], desc="正在整合 LLM 判斷結果"):
        vegetarian = llm_vege_results.get(rid, False)
        # LLM判斷為葷食時，預設uses_pork為false，因為關鍵字比對階段沒找到
        uses_pork = False 
        
        result = {"vegetarian": vegetarian, "uses_pork": uses_pork}
        if ENABLE_REASON_LOGGING:
            result["reason"] = "LLM_classified"
        final_results[rid] = result

    # 存 JSON
    out_dir = "data/embeddings"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Meat_and_Vegetarian_final_safe.json")

    with open(out_path, "w", encoding="utf-8") as jf:
        json.dump(final_results, jf, ensure_ascii=False, indent=2)

    print(f"\n分類完成！JSON 生成位置：{os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()