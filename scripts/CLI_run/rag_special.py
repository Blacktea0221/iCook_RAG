import sys, os
import json
import random
import jieba
from dotenv import load_dotenv

# 專案路徑加入 Python 模組搜尋
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# === Google 搜尋（備援，暫關） ===
from googlesearch import search as google_search  # pip install googlesearch-python

# === LLM 與資料庫工具 ===
from scripts.RAG.llm_utils import call_ollama_llm, summarize_search_results
from scripts.database.ingredient_utils import build_ingredient_set_from_db
from scripts.RAG.search_engine import get_recipe_by_id, pull_ingredients, tag_then_vector_rank

load_dotenv()

# === 設定檔案路徑 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
classify_map_path = os.path.join(project_root, "data", "embeddings", "Meat and Vegetarian.json")

with open(classify_map_path, "r", encoding="utf-8") as f:
    CLASSIFY_MAP = json.load(f)

CLASS_DICT = {"素食", "葷食"}
CLASS_MAPPING = {"素食": "vegetarian", "葷食": "non_vegetarian"}


# === Google 搜尋食譜備援函式 ===
def google_search_recipes(keyword: str, k: int = 5):
    query = f"{keyword} 食譜"
    results = []
    for item in google_search(query, advanced=True, num_results=k, lang="zh-tw"):
        results.append({"title": item.title, "link": item.url, "snippet": item.description})
    return results


# === CLI 顯示食譜函式 ===
def pretty_print(item: dict):
    rec = item["recipe"]
    print(f"=== 查詢結果：Recipe ID {item['id']} (相似度 {item.get('score', 1.0):.4f}) ===\n")
    print(f"食譜名稱：{rec.get('recipe','')}\n")

    if rec.get("preview_tags"):
        print("── 預覽 Tags ──")
        print("、".join(rec["preview_tags"]))
        print()

    print("── 食材 Ingredients ──")
    for idx, ing in enumerate(rec.get("ingredients", []), 1):
        print(f"{idx}. {ing.get('ingredient','')}")

    print("\n── 步驟 Steps ──")
    for step in rec.get("steps", []):
        print(f"{step.get('step_no','')}. {step.get('description','')}")
    print()


# === 主程式 ===
def main():
    print("RAG 智能推薦查詢（輸入任何中文描述；exit 離開）")

    # 初始化食材集合
    try:
        ingredient_set = build_ingredient_set_from_db()
    except Exception as e:
        print("[FATAL] 初始化食材字典失敗：", repr(e))
        return

    while True:
        raw_input_text = input("\n請描述你有的食材或需求: ").strip()
        if raw_input_text.lower() in ("exit", "quit"):
            print("離開程式")
            break

        # === 特殊飲食條件判斷 ===
        classes = [t for t in CLASS_DICT if t in raw_input_text]
        hates_pork = ("不吃豬肉" in raw_input_text) or ("無豬" in raw_input_text)

        # 先從 CLASSIFY_MAP 過濾符合飲食條件的食譜 ID
        allowed_ids = None
        if classes:
            diet_key = CLASS_MAPPING[classes[0]]  # vegetarian / non_vegetarian
            allowed_ids = [
                int(rid) for rid, info in CLASSIFY_MAP.items() if info.get(diet_key, False)
            ]

        if hates_pork and allowed_ids:
            allowed_ids = [
                rid for rid in allowed_ids if not CLASSIFY_MAP.get(str(rid), {}).get("uses_pork", False)
            ]

        # === 抽取關鍵字 ===
        keywords = pull_ingredients(raw_input_text, ingredient_set)

        # 指定飲食分類但無關鍵字 → 隨機推薦
        if classes and not keywords and allowed_ids:
            sample_ids = random.sample(allowed_ids, k=min(3, len(allowed_ids)))
            for rid in sample_ids:
                rec = get_recipe_by_id(rid)
                if rec:
                    pretty_print({"id": rid, "score": 1.0, "recipe": rec})
            continue

        # 抽不到可用關鍵字 → Google 備援
        if not keywords:
            print("⚠️ 未偵測到任何可用食材，改為網路搜尋模式…")
            hits = google_search_recipes(raw_input_text, k=5)
            summary = summarize_search_results(raw_input_text, hits)
            print(summary)
            continue

        # === 核心排序：Tag + 向量排序（僅在 allowed_ids 範圍） ===
        results = tag_then_vector_rank(
            user_text=raw_input_text,
            tokens_from_jieba=keywords,
            top_k=5,
            candidate_ids=allowed_ids  # 只對符合飲食條件的食譜排序
        )

        if not results:
            print("⚠️ 本地資料庫查無結果，嘗試網路搜尋…")
            hits = google_search_recipes(raw_input_text, k=5)
            summary = summarize_search_results(raw_input_text, hits)
            print(summary)
            continue

        print("\n正在自動推薦最適合的食譜...\n")
        try:
            summary = call_ollama_llm(raw_input_text, results)
            print(summary)
        except Exception as e:
            print("[WARN] LLM 摘要失敗：", repr(e))
            continue

        # 詢問是否查看詳細食譜
        view_choice = input("\n是否查看詳細食譜？輸入 y 查看 / n 跳過：").strip().lower()
        if view_choice == "y":
            id_input = input("請輸入要查看的食譜 ID：").strip()
            if id_input.isdigit():
                rec = get_recipe_by_id(id_input)
                if rec:
                    pretty_print({"id": id_input, "score": 1.0, "recipe": rec})
                else:
                    print("❌ 查無此 ID 的食譜")
            else:
                print("❌ ID 必須是數字")


if __name__ == "__main__":
    main()
