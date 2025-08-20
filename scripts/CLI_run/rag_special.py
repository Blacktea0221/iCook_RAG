import sys
import os
import json
import random
import jieba
from typing import List, Dict, Any, Set


# 取得目前檔案的絕對路徑
current_file_path = os.path.abspath(__file__)
# 取得專案根目錄路徑
project_root_path = os.path.abspath(
    os.path.join(os.path.dirname(current_file_path), "../..")
)
# 將專案根目錄加入 Python 模組搜尋路徑
sys.path.insert(0, project_root_path)


# 從其他模組中匯入所需的函式
from scripts.RAG.llm_utils import call_openai_llm
from scripts.RAG.search_engine import (
    get_recipe_by_id,
    tag_then_vector_rank,
    pull_ingredients,
    fetch_all,
)


# 定義分類標籤和其對應的關鍵字
DIET_KEYWORDS = {
    "素食": "vegetarian",
    "葷食": "non_vegetarian",
    "不吃豬肉": "uses_pork",
    "無豬": "uses_pork",
    "穆斯林": "uses_pork",
}


# 新增：定義豬肉製品相關的關鍵字集合
PORK_PRODUCTS = {
    "豬肉",
    "豬油",
    "培根",
    "火腿",
    "香腸",
    "豬腳",
    "豬尾巴",
    "豬耳朵",
    "豬腸",
    "豬排",
    "豬里脊",
    "豬腿肉",
    "豬腰",
    "豬肝",
    "豬心",
    "豬肋排",
    "豬五花",
    "豬絞肉",
    "豬排骨",
    "豬油",
    "貢丸",
}

# 修正：確保 ingredient_set 被正確定義
ingredient_set = {
    "番茄",
    "雞蛋",
    "豬肉",
    "牛肉",
    "蔥",
    "豆腐",
    "花椒",
    "醬油",
    "蒜苗",
    "豬油",
    "韭菜",
    "培根",
    "火腿",
    "香腸",
    "豬腳",
}


# === 資料載入函式 ===
def _load_dietary_groups_from_db() -> Dict[str, Dict[str, bool]]:
    """
    從資料庫的 dietary_groups 表格中載入飲食群組資料。
    """
    print("正在從資料庫載入 dietary_groups...")
    try:
        sql = "SELECT recipe_id, vegetarian, uses_pork FROM dietary_groups;"
        rows = fetch_all(sql)
        # 轉換資料格式
        classify_map = {}
        for r in rows:
            classify_map[str(r["recipe_id"])] = {
                "vegetarian": r["vegetarian"],
                "uses_pork": r["uses_pork"],
            }
        print(f"成功載入 {len(classify_map)} 筆食譜的飲食群組資料。")
        return classify_map
    except Exception as e:
        print(f"❌ 從資料庫載入 dietary_groups 失敗: {e}")
        return {}


# 在模組載入時，將資料庫資料載入到一個全域變數中，只執行一次
CLASSIFY_MAP = _load_dietary_groups_from_db()


# === 核心功能：3-2 group 相關檢索函式 ===
def group_retrieval(raw_input_text: str, top_k: int = 3):
    """
    執行 3-2 group 相關的檢索流程。
    """
    print(f"處理使用者輸入: '{raw_input_text}'")

    if not CLASSIFY_MAP:
        return json.dumps(
            {"error": "無法連線至資料庫，請稍後再試。"}, ensure_ascii=False, indent=2
        )

    # 1. 偵測所有飲食關鍵字
    is_vegetarian = any(kw in raw_input_text for kw in ["素食"])
    is_non_vegetarian = any(kw in raw_input_text for kw in ["葷食"])
    is_no_pork = any(kw in raw_input_text for kw in ["不吃豬肉", "無豬", "穆斯林"])

    # 偵測關鍵字，用來處理衝突
    keywords = pull_ingredients(raw_input_text, ingredient_set)
    # 修正：用 PORK_PRODUCTS 集合判斷是否包含任何豬肉製品
    contains_pork_ingr = any(k in keywords for k in PORK_PRODUCTS)
    contains_beef_ingr = "牛肉" in keywords

    # 2. 處理嚴謹條件衝突
    # 2-1) 素食優先於任何肉類
    if is_vegetarian and (contains_pork_ingr or contains_beef_ingr):
        print("偵測到素食與肉類衝突，優先考慮素食，忽略肉類關鍵字。")
        # 修正：使用 PORK_PRODUCTS 集合來排除關鍵字
        keywords = [k for k in keywords if k not in PORK_PRODUCTS and k != "牛肉"]
    # 2-2) 葷食與素食衝突
    if is_vegetarian and is_non_vegetarian:
        return json.dumps(
            {"error": "素食與葷食條件衝突，請明確選擇一種。"},
            ensure_ascii=False,
            indent=2,
        )
    # 2-3) 穆斯林(無豬)優先於豬肉
    if is_no_pork and contains_pork_ingr:
        print("偵測到無豬肉與豬肉關鍵字衝突，優先考慮無豬肉，忽略豬肉關鍵字。")
        # 修正：使用 PORK_PRODUCTS 集合來排除關鍵字
        keywords = [k for k in keywords if k not in PORK_PRODUCTS]

    # 3. 建立所有食譜 ID 的集合作為篩選起點
    all_recipe_ids = set(CLASSIFY_MAP.keys())
    candidate_ids: Set[str] = set()

    # 4. 依序應用過濾器
    if is_vegetarian:
        candidate_ids = {
            rid for rid, info in CLASSIFY_MAP.items() if info.get("vegetarian", False)
        }
    elif is_non_vegetarian:
        candidate_ids = {
            rid
            for rid, info in CLASSIFY_MAP.items()
            if not info.get("vegetarian", False)
        }

    if not candidate_ids:
        candidate_ids = all_recipe_ids

    if is_no_pork:
        pork_free_ids = {
            rid
            for rid, info in CLASSIFY_MAP.items()
            if not info.get("uses_pork", False)
        }
        candidate_ids = candidate_ids.intersection(pork_free_ids)

    # 5. 處理特殊情況：只指定 group 但無關鍵字
    if (is_vegetarian or is_non_vegetarian or is_no_pork) and not keywords:
        print("偵測到特殊飲食條件，但沒有關鍵食材，隨機推薦符合條件的食譜...")

        if not candidate_ids:
            return json.dumps(
                {"error": "無法找到符合所有飲食條件的食譜。請嘗試放寬條件。"},
                ensure_ascii=False,
                indent=2,
            )

        sample_ids = random.sample(
            list(candidate_ids), k=min(top_k, len(candidate_ids))
        )

        results = []
        for rid in sample_ids:
            rec = get_recipe_by_id(str(rid))
            if rec:
                results.append({"id": rid, "score": 1.0, "recipe": rec})

        summary_str = call_openai_llm(raw_input_text, results)
        return summary_str

    # 6. 核心檢索：Tag + 向量排序
    if not keywords:
        return json.dumps(
            {"error": "未偵測到任何可用關鍵字，請嘗試更具體的描述。"},
            ensure_ascii=False,
            indent=2,
        )

    print("正在對符合條件的食譜進行向量排序...")

    results = tag_then_vector_rank(
        user_text=raw_input_text,
        tokens_from_jieba=keywords,
        candidate_ids=list(candidate_ids),
        top_k=top_k,
    )

    if not results:
        return json.dumps({"error": "查無結果。"}, ensure_ascii=False, indent=2)

    # 7. 將結果傳給 LLM 進行整理與輸出
    try:
        summary_str = call_openai_llm(raw_input_text, results)
        return summary_str
    except Exception as e:
        return json.dumps({"error": f"LLM 摘要失敗: {e}"}, ensure_ascii=False, indent=2)


# === 測試區塊 ===
def test_group_retriever():
    """提供一些預設輸入進行測試"""
    print("--- 執行 3-2 群組檢索功能自動測試 ---")

    test_cases = [
        # "素食食譜，要有番茄",
        # "穆斯林想做蔥爆牛肉",
        # "不吃豬肉，我想要蔥爆牛肉",
        # 蒜末 蒜頭
        "我想要素食",
        # 豬絞肉 豬肉片 梅花豬肉片
        "我想要穆斯林食譜",
        "素食食譜，但我想吃牛肉",
        "我要素食，要有洋蔥和豆腐",
        "我要穆斯林，想要貢丸和醬油",
        "我想要素食，但我想吃雞蛋",
        "我要素食，有花椰菜胡蘿蔔",
    ]

    for case in test_cases:
        print(f"\n[測試案例] 輸入: '{case}'")
        summary_str = group_retrieval(case)

        try:
            summary_json = json.loads(summary_str)
            print(
                "\n[AI 推薦摘要]\n"
                + json.dumps(summary_json, ensure_ascii=False, indent=2)
            )
        except (json.JSONDecodeError, TypeError):
            print("\n[AI 推薦摘要]\n" + summary_str)

        print("-" * 50)


# 僅在直接執行此檔案時執行測試
if __name__ == "__main__":
    test_group_retriever()
