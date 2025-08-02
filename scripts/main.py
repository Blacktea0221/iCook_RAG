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
    # 1. 載入所有資料與配置
    data = load_data()
    df_cleaned = data["df_cleaned"]
    df_preview = data["df_preview"]
    df_detailed = data["df_detailed"]
    df_steps = data["df_steps"]
    ingredient_set = data["ingredient_set"]
    id2tags = data["id2tags"]
    CLASSIFY_MAP = data["classify_map"]

    CLASS_DICT = {"素食", "葷食"}
    CLASS_MAPPING = {"素食": "vegetarian", "葷食": "non_vegetarian"}

    print("RAG 智能推薦查詢（輸入任何中文描述；exit 離開）")

    while True:
        raw_input_text = input("\n請描述你有的食材或需求: ").strip()
        if raw_input_text.lower() in ("exit", "quit"):
            break

        # 1) 用 Jieba 切詞（如果有需求）
        # tokens = jieba.lcut(raw_input_text)

        # 2) 抽出 class、hates_pork
        classes = [t for t in CLASS_DICT if t in raw_input_text]
        hates_pork = "不吃豬肉" in raw_input_text

        # 3) 先擷取食材關鍵字（Jieba為主，失敗可接llm_utils，但此例只呼叫 search_engine）
        keywords = pull_ingredients(raw_input_text, ingredient_set)

        # 4) diet 與 pork 過濾 → allowed_ids
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

        # 5) 只有輸入 class（如「素食」）沒有 keywords，就隨機顯示 3 道
        if classes and not keywords:
            sample_ids = random.sample(allowed_ids, k=min(3, len(allowed_ids)))
            for rid in sample_ids:
                rec = get_recipe_by_id(
                    rid, df_cleaned, df_preview, df_detailed, df_steps
                )
                pretty_print({"id": rid, "score": 1.0, "recipe": rec})
            continue

        # 6) 沒有任何 keywords，就跑 Google 備援
        if not keywords:
            print("⚠️ 未偵測到任何可用食材，改為網路搜尋模式…")
            web_hits = google_search_recipes(raw_input_text, k=5)
            if not web_hits:
                print("🚫 Google 無結果，請嘗試其他關鍵字。")
                continue
            summary = summarize_search_results(raw_input_text, web_hits)
            print("🌐 來自 Google 的推薦：\n" + summary + "\n")
            continue

        # 7) 有關鍵字 → 本地檢索
        query = ", ".join(keywords)
        res = search_by_partial_ingredients(
            query,
            id2tags,
            data["model"],
            data["embeddings"],
            data["emb_norms"],
            data["tags"],
            df_cleaned,
            df_preview,
            df_detailed,
            df_steps,
            top_k=3,
            allowed_ids=allowed_ids,
        )
        # 7.1) 如果使用者有說 "不吃豬肉"，再剔除所有 uses_pork = True 的項目
        if hates_pork:
            res = [hit for hit in res if not CLASSIFY_MAP[str(hit["id"])]["uses_pork"]]

        # 8) 若本地查無結果，再跑 Google
        if not res:
            print("⚠️ 本地資料庫查無結果，嘗試網路搜尋…")
            web_hits = google_search_recipes(query, k=5)
            if not web_hits:
                print("🚫 Google 無結果，請嘗試其他關鍵字。")
                continue
            summary = summarize_search_results(query, web_hits)
            print("🌐 來自 Google 的推薦：\n" + summary + "\n")
            continue

        print("\n正在自動推薦最適合的食譜...\n")
        answer = call_ollama_llm(query, res)
        print("🧠 智能推薦：\n" + answer + "\n")

        print(
            "🔍 若想查看其中一道食譜的【詳細食材與步驟】，"
            "請輸入該食譜『名稱關鍵字』或該食譜的 ID"
        )
        print("✏️ 若想重新查詢其他食材，請輸入 new；離開請輸入 exit。")

        name_map = {r["recipe"]["食譜名稱"]: r["id"] for r in res}
        id_set = set(r["id"] for r in res)
        selected_id = None

        while True:
            follow_up = input(
                "請輸入想查看詳情的食譜編號/名稱，或輸入 new 查詢新食材，也可以輸入exit 退出 "
            ).strip()
            if follow_up.lower() in ("exit", "quit"):
                exit()
            if follow_up.lower() in ("new", ""):
                break

            if follow_up.isdigit() and int(follow_up) in id_set:
                selected_id = int(follow_up)
            else:
                for name, rid in name_map.items():
                    if follow_up in name:
                        selected_id = rid
                        break

            if selected_id:
                recipe = get_recipe_by_id(
                    selected_id, df_cleaned, df_preview, df_detailed, df_steps
                )
                if recipe:
                    pretty_print({"id": selected_id, "score": 1.0, "recipe": recipe})
                    print(
                        "\n📌 您可以輸入其他 ID 或名稱繼續查看，或輸入 new 查詢新內容。"
                    )
                else:
                    print("找不到該食譜的詳細資訊。")
            else:
                print("無法辨識輸入內容，請再輸入一次。")


if __name__ == "__main__":
    main()
