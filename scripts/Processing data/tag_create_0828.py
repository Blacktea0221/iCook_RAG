import os
import json
from dotenv import load_dotenv
import openai
from tqdm import tqdm
import math

# --- 載入環境變數 ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- 定義關鍵字列表 ---
# 這些關鍵字用於第一階段的快速比對。
# 五辛、肉類和海鮮被視為葷食。
MEAT_INGREDIENTS = [
    "豬肉", "豬排", "五花肉", "絞肉", "排骨", "培根", "火腿", "香腸", "臘肉", "貢丸",
    "豬肋排", "豬里脊", "豬腿肉", "豬腰", "豬肝", "豬心", "豬腳", "豬耳朵", "豬尾巴",
    "雞肉", "雞腿", "雞翅", "牛肉", "牛排", "羊肉", "魚", "蝦", "螃蟹", "蛤蜊",
    "花枝", "干貝", "小魚乾", "魚板", "魚漿", "章魚", "肉燥", "肉鬆", "雞胸肉",
    "雞塊", "鴨肉", "鵝肉", "豬舌", "豬肚", "豬血", "雞胗", "鴨血", "鴨腸", "雞心",
    "鵝肝", "鮑魚", "海參", "魚翅", "魚肚", "燕窩", "魷魚", "鮮蝦", "蚵仔",
    "生蠔", "淡菜", "鯖魚", "鮭魚", "鮪魚", "鯛魚", "秋刀魚", "虱目魚", "土雞肉",
    "烏骨雞", "鵝肉", "鴨腸", "牛筋", "牛肚", "牛腱", "羊肉串", "羊排", "羊腩",
    "兔肉", "鹿肉", "豬油渣", "蝦米", "蝦皮", "魚卵", "蟹肉", "蟹膏", "飛魚卵",
    "烏魚子", "干貝醬", "魚露", "蝦醬", "沙茶醬", "雞精", "雞粉", "雞湯塊", "豬油",
    "牛油", "肉骨湯", "豬肉片", "牛肉片", "羊肉片", "雞肉塊", "牛舌", "牛尾", "豬頭皮",
    "豬大腸", "豬小腸", "豬肺", "雞冠", "雞皮", "鴨掌", "鴨舌", "鯊魚", "龍蝦", "田雞",
    "海蜇皮", "海膽", "扇貝", "螺肉", "鮪魚罐頭", "鯖魚罐頭", "魚鬆", "雞肉丸", "豬肉丸",
    "魚丸", "蝦丸", "花枝丸", "獅子頭", "德國香腸", "熱狗", "火腿片", "培根片", "雞柳",
    "豬絞肉", "牛肉燥", "豬肉條", "雞脖子", "鴨脖子", "鵝掌", "鵝頭", "豬腳筋",
    "牛骨", "雞骨", "魚骨", "肉骨", "雞湯", "魚高湯", "蝦高湯", "肉高湯", "牛高湯",
    "豬骨湯", "老母雞湯", "火雞肉", "鴿肉", "海鮮醬", "海鮮粉", "XO醬", "蠔油",
    "鰻魚醬", "鰹魚粉", "柴魚片", "柴魚粉", "烤肉醬", "肉骨茶", "三層肉",
    "蔥", "青蔥", "大蔥", "小蔥", "細香蔥", "紅蔥頭", "洋蔥", "黃洋蔥", "白洋蔥",
    "紫洋蔥", "蒜", "大蒜", "蒜頭", "蒜瓣", "蒜泥", "蒜末", "韭菜", "韭菜花",
    "韭黃", "薤", "五辛"
]
PORK_INGREDIENTS = [
    "豬肉", "豬排", "五花肉", "絞肉", "排骨", "培根", "火腿", "香腸", "臘肉", "貢丸",
    "豬肋排", "豬里脊", "豬腿肉", "豬腰", "豬肝", "豬心", "豬腳", "豬耳朵", "豬尾巴",
    "豬舌", "豬肚", "豬血", "豬油渣", "豬肉片", "豬頭皮", "豬大腸", "豬小腸", "豬肺",
    "豬絞肉", "豬肉條", "豬骨", "肉骨", "豬骨湯", "豬肉丸", "獅子頭", "德國香腸",
    "熱狗", "火腿片", "培根片", "豬腳筋", "肉骨茶湯料", "豬油", "豬骨湯塊", "豬骨高湯",
    "豬骨粉", "豬肉粉", "豬肉精粉", "豬肉湯塊", "豬油渣粉", "豬骨濃縮液", "豬骨精"
]

def get_llm_judgment(ingredients):
    """
    呼叫 OpenAI API，並使用 Few-shot 範例進行判斷。
    """
    ingredients_str = ", ".join(ingredients)
    
    # Few-shot Prompt
    prompt = f"""
    請作為一個食譜分析專家，根據提供的食材清單，判斷食譜的屬性。
    請嚴格遵守以下規則：
    1. 素食的定義為「全素」及「蛋奶素」，因此含蛋與奶製品的食譜仍視為素食。
    2. 如果食譜含有任何肉類、魚類、海鮮或五辛（蔥、蒜、韭菜、洋蔥、薤），"vegetarian" 應為 false。
    3. 如果食譜含有豬肉或任何豬肉製品，"uses_pork" 應為 true。
    4. 最終結果必須以 {{ "vegetarian": true/false, "uses_pork": true/false }} 的 JSON 格式輸出，不要包含任何其他文字或解釋。

    以下是一些範例：
    ---
    食材: 雞蛋, 青蔥, 鹽, 白胡椒粉
    JSON: {{"vegetarian": false, "uses_pork": false}}
    ---
    食材: 火腿, 起司片, 牛奶, 吐司
    JSON: {{"vegetarian": false, "uses_pork": true}}
    ---
    食材: 豆干, 芹菜, 紅蘿蔔
    JSON: {{"vegetarian": true, "uses_pork": false}}
    ---
    食材: 豬肉, 白菜, 蒜頭
    JSON: {{"vegetarian": false, "uses_pork": true}}
    ---
    食材: 綠竹筍, 蒜頭, 辣椒, 蔥
    JSON: {{"vegetarian": false, "uses_pork": false}}
    ---
    食材: 雞蛋, 馬鈴薯, 番茄醬
    JSON: {{"vegetarian": true, "uses_pork": false}}
    ---
    現在，請為以下食材清單提供判斷結果：
    食材: {ingredients_str}
    JSON:
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes recipes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        raw_text = response.choices[0].message.content.strip()
        return json.loads(raw_text)
    except Exception as e:
        print(f"呼叫 LLM 錯誤: {e}")
        return None

def process_recipes(input_file, output_file, mode="test", limit=50, offset=0):
    """
    處理食譜資料，進行關鍵字比對和 LLM 驗證。
    - mode: "test" 進行測試模式，"full" 處理所有資料。
    - limit & offset: 測試模式下的資料範圍。
    """
    results = {}

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {input_file}")
        return

    recipe_ids = list(recipes_data.keys())
    if mode == "test":
        end_index = min(offset + limit, len(recipe_ids))
        recipe_ids = recipe_ids[offset:end_index]
        print(f"測試模式: 處理從第 {offset} 筆開始的 {len(recipe_ids)} 筆資料。")

    for recipe_id in tqdm(recipe_ids, desc="正在處理食譜"):
        ingredients = recipes_data[recipe_id]
        ingredients_str = " ".join(ingredients)
        
        # 第一層判斷：關鍵字比對
        is_vegetarian_keyword = not any(keyword in ingredients_str for keyword in MEAT_INGREDIENTS)
        uses_pork_keyword = any(keyword in ingredients_str for keyword in PORK_INGREDIENTS)
        
        # 第二層：LLM 最終驗證
        llm_result = get_llm_judgment(ingredients)
        
        if llm_result:
            results[recipe_id] = llm_result
        else:
            results[recipe_id] = {
                "vegetarian": is_vegetarian_keyword,
                "uses_pork": uses_pork_keyword
            }
            print(f"LLM 驗證失敗，退回關鍵字判斷結果: {results[recipe_id]}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n處理完成，結果已儲存至 {os.path.abspath(output_file)}")

# --- 程式入口點 ---
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(os.path.dirname(BASE_DIR))
    input_json_path = os.path.join(project_root_dir, "data", "embeddings", "ingredient_list.json")
    
    output_json_path = os.path.join(BASE_DIR, "processed_recipes_test_02.json")
    
    # === 測試模式 ===
    # process_recipes(input_json_path, output_json_path, mode="test", limit=50, offset=1500)
    
    # === 正式模式（註解掉以避免誤觸） ===
    output_json_path_full = os.path.join(BASE_DIR, "processed_recipes_full.json")
    process_recipes(input_json_path, output_json_path_full, mode="full")