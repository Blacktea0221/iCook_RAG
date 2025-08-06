import json
import random

# Google search 邏輯可自行包一個 utils/ 或 inline
from googlesearch import search as google_search  # pip install googlesearch-python
from RAG.data_loader import load_data
from RAG.llm_utils import call_ollama_llm, summarize_search_results
from RAG.search_engine import (
    get_recipe_by_id,
    pull_ingredients,
    search_by_partial_ingredients,
)

# 只讀取本地 classify_map
with open("data/embeddings/Meat and Vegetarian.json", "r", encoding="utf-8") as f:
    CLASSIFY_MAP = json.load(f)

# 這裡你可以載入 ingredient_set（之前是 build_ingredient_set 的結果，建議還是先用本地檔案匯入成 set）

ingredient_set = set()  # ← TODO: 你可以加載或寫一個 build_ingredient_set_from_db

CLASS_DICT = {"素食", "葷食"}
CLASS_MAPPING = {"素食": "vegetarian", "葷食": "non_vegetarian"}


def google_search_recipes(keyword: str, k: int = 5):
    query = f"{keyword} 食譜"
    results = []
    for item in google_search(query, advanced=True, num_results=k, lang="zh-tw"):
        results.append(
            {"title": item.title, "link": item.url, "snippet": item.description}
        )
    return results


def pretty_print(item: dict):
    """簡化版結果輸出"""
    rec = item["recipe"]
    print(
        f"=== 查詢結果：Recipe ID {item['id']} (相似度 {item.get('score',1.0):.4f}) ===\n"
    )
    print(
        f"食譜名稱：{rec.get('食譜名稱','')}\n"
        f"分類　　　：{rec.get('vege_name','')}\n"
    )
    print("── 食材 Ingredients ──")
    for idx, ing in enumerate(rec.get("ingredients", []), 1):
        print(
            f"{idx}. {ing.get('ingredient_name','')} {ing.get('quantity','')}{ing.get('unit','')}"
        )
    print()
    print("── 步驟 Steps ──")
    for step in rec.get("steps", []):
        print(f"{step.get('step_no','')}. {step.get('description','')}")
    print()


def main():
    print("RAG 智能推薦查詢（輸入任何中文描述；exit 離開）")
    while True:
        raw_input_text = input("\n請描述你有的食材或需求: ").strip()
        if raw_input_text.lower() in ("exit", "quit"):
            break

        classes = [t for t in CLASS_DICT if t in raw_input_text]
        hates_pork = "不吃豬肉" in raw_input_text

        keywords = pull_ingredients(raw_input_text, ingredient_set)

        allowed_ids = None
        if classes:
            diet_key = CLASS_MAPPING[classes[0]]
            allowed_ids = [
                int(rid)
                for rid, info in CLASSIFY_MAP.items()
                if info["diet"] == diet_key
            ]
            if hates_pork:
                allowed_ids = [
                    rid
                    for rid in allowed_ids
                    if not CLASSIFY_MAP[str(rid)]["uses_pork"]
                ]

        if classes and not keywords and allowed_ids:
            sample_ids = random.sample(allowed_ids, k=min(3, len(allowed_ids)))
            for rid in sample_ids:
                rec = get_recipe_by_id(rid)
                pretty_print({"id": rid, "score": 1.0, "recipe": rec})
            continue

        if not keywords:
            print("⚠️ 未偵測到任何可用食材，改為網路搜尋模式…")
            # 保留你的 Google 搜尋邏輯
            continue

        query = ", ".join(keywords)
        res = search_by_partial_ingredients(
            query,
            top_k=3,
            allowed_ids=allowed_ids,
        )
        if hates_pork:
            res = [hit for hit in res if not CLASSIFY_MAP[str(hit["id"])]["uses_pork"]]

        if not res:
            print("⚠️ 本地資料庫查無結果，嘗試網路搜尋…")
            # 保留你的 Google 搜尋邏輯
            continue

        print("\n正在自動推薦最適合的食譜...\n")
        # 你的 call_ollama_llm 保持不變
        # pretty_print 等細節保持不變


if __name__ == "__main__":
    main()
