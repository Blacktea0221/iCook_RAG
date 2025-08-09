# main.py
import json
import random

import jieba
from dotenv import load_dotenv

# === 保留：Google 搜尋（已關閉實際呼叫） ===
from googlesearch import search as google_search  # pip install googlesearch-python

from scripts.RAG.llm_utils import call_ollama_llm, summarize_search_results

# === 改：不再用 data_loader，直接用 search_engine 提供的 DB utils 與檢索 ===
from scripts.RAG.search_engine import fetch_all  # 新增：直接拿 DB 資料
from scripts.RAG.search_engine import (
    get_recipe_by_id,
    pull_ingredients,
    tag_then_vector_rank,
)

load_dotenv()

# 只讀取本地 classify_map（沿用你原本的設定）
with open("data/embeddings/Meat and Vegetarian.json", "r", encoding="utf-8") as f:
    CLASSIFY_MAP = json.load(f)

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


def build_ingredient_set_from_db() -> set:
    """
    直接從資料庫建立 ingredient_set：抓 ingredient + preview_tag 去重。
    表：public.ingredient（recipe_id, ingredient, preview_tag）
    """
    rows = fetch_all(
        "SELECT ingredient, preview_tag FROM public.ingredient WHERE ingredient IS NOT NULL OR preview_tag IS NOT NULL;"
    )
    words = set()
    for r in rows:
        ing = (r.get("ingredient") or "").strip()
        tag = (r.get("preview_tag") or "").strip()
        if ing:
            words.add(ing)
        if tag:
            words.add(tag)
    # 讓 jieba 能切出你資料庫裡的詞
    for w in words:
        if w:
            jieba.add_word(w)
    print(f"[init] 食材字典已初始化（共 {len(words)} 項）")
    return words


def pretty_print(item: dict):
    """輸出單筆檢索結果（配合新版 get_recipe_by_id 的結構）"""
    rec = item["recipe"]
    print(
        f"=== 查詢結果：Recipe ID {item['id']} (相似度 {item.get('score', 1.0):.4f}) ===\n"
    )
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


def main():
    print("RAG 智能推薦查詢（輸入任何中文描述；exit 離開）")

    # === 直接從 DB 初始化 ingredient_set（不再依賴 data_loader） ===
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

        # ========== 特殊飲食條件 ==========
        classes = [t for t in CLASS_DICT if t in raw_input_text]
        hates_pork = ("不吃豬肉" in raw_input_text) or ("無豬" in raw_input_text)

        allowed_ids = None
        if classes:
            diet_key = CLASS_MAPPING[classes[0]]
            allowed_ids = [
                int(rid)
                for rid, info in CLASSIFY_MAP.items()
                if info.get("diet") == diet_key
            ]
            if hates_pork:
                allowed_ids = [
                    rid
                    for rid in allowed_ids
                    if not CLASSIFY_MAP.get(str(rid), {}).get("uses_pork", False)
                ]

        # ========== 關鍵字抽取 ==========
        keywords = pull_ingredients(raw_input_text, ingredient_set)

        # 僅指定飲食分類、沒關鍵字：隨機推薦
        if classes and not keywords and allowed_ids:
            sample_ids = random.sample(allowed_ids, k=min(3, len(allowed_ids)))
            for rid in sample_ids:
                rec = get_recipe_by_id(rid)
                if rec:
                    pretty_print({"id": rid, "score": 1.0, "recipe": rec})
            continue

        # 抽不到可用關鍵字：走 Google 備援（暫關）
        if not keywords:
            print("⚠️ 未偵測到任何可用食材，改為網路搜尋模式…")
            hits = google_search_recipes(raw_input_text, k=5)
            summary = summarize_search_results(raw_input_text, hits)
            print(summary)
            continue

        # ========== 核心：先 Tag 候選，再整句向量排序 ==========
        results = tag_then_vector_rank(
            user_text=raw_input_text,
            tokens_from_jieba=keywords,
            top_k=5,
        )

        # 飲食與豬肉過濾（若指定）
        if allowed_ids is not None:
            allowed_set = set(allowed_ids)
            results = [r for r in results if int(r["id"]) in allowed_set]
        if hates_pork:
            results = [
                r
                for r in results
                if not CLASSIFY_MAP.get(str(r["id"]), {}).get("uses_pork", False)
            ]

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

        # 詢問是否查看詳細
        view_choice = (
            input("\n是否查看詳細食譜？輸入 y 查看 / n 跳過：").strip().lower()
        )
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
