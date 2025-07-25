#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_and_retrieve_recipes.py

整合文字檢索與完整食譜回傳功能，並以易讀格式輸出：
1. 載入 embeddings 與 metadata
2. 載入清理後的食譜資料
3. 提供 search_similar(query, top_k) 與 search_and_retrieve 函式
4. CLI 互動式輸入，並以自訂格式列印結果（不含預覽標籤）

使用前請安裝依賴：
$ pip install pandas numpy sentence-transformers

執行：
$ python search_and_retrieve_recipes.py
"""
import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
# from integrate_recipes import get_recipe_by_id
from sentence_transformers import SentenceTransformer

# 取得當前腳本目錄及上層目錄
# 可調整分類
category = "九層塔"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

tags_path   = os.path.join(ROOT_DIR, "data", "embeddings", category, "tags.json")
embed_path  = os.path.join(ROOT_DIR, "data", "embeddings", category, "embeddings.npy")
cleaned_path   = os.path.join(ROOT_DIR, "data", "clean", category, f"{category}_recipes_cleaned.csv")
preview_path   = os.path.join(ROOT_DIR, "data", "clean", category, f"{category}_preview_ingredients.csv")
detailed_path  = os.path.join(ROOT_DIR, "data", "clean", category, f"{category}_detailed_ingredients.csv")
steps_path     = os.path.join(ROOT_DIR, "data", "clean", category, f"{category}_recipe_steps.csv")


# 一次載入 embeddings、tags、模型
with open(tags_path, "r", encoding="utf-8") as f:
    tags = json.load(f)
embeddings = np.load(embed_path)
model = SentenceTransformer("BAAI/bge-m3")
emb_norms = np.linalg.norm(embeddings, axis=1)

# 建立 id2tags: id -> set(tag)
id2tags = defaultdict(set)
for item in tags:
    id2tags[item["id"]].add(item["tag"])

# 載入食譜資料
print("載入清理後的食譜資料...")
# recipes_cleaned.csv 使用分號分隔
try:
    df_cleaned = pd.read_csv(cleaned_path, sep=";", encoding="utf-8-sig")
    df_cleaned.columns = df_cleaned.columns.str.strip()
except Exception as e:
    print(f"讀取 {cleaned_path} 失敗：{e}")
    raise

# 其他 CSV 為逗號分隔
try:
    df_preview = pd.read_csv(preview_path, encoding="utf-8-sig").rename(
        columns=lambda x: x.strip()
    )
    df_detailed = pd.read_csv(detailed_path, encoding="utf-8-sig").rename(
        columns=lambda x: x.strip()
    )
    df_steps = pd.read_csv(steps_path, encoding="utf-8-sig").rename(
        columns=lambda x: x.strip()
    )
except Exception as e:
    print(f"讀取預覽/詳細/步驟檔案失敗：{e}")
    raise

def get_recipe_by_id(recipe_id, dfs):
    """
    根據 recipe_id 從已載入的 DataFrame 中取得完整食譜資料。
    返回一個 dict 包括：
      - 主表欄位
      - preview_tags（list）
      - ingredients（list of dict）
      - steps（list of dict）
    """
    df_recipes, df_preview, df_detailed, df_steps = dfs

    # 取出食譜主表的一筆資料
    rec = df_recipes[df_recipes["id"] == recipe_id]
    if rec.empty:
        return None
    rec_dict = rec.iloc[0].to_dict()

    # 預覽食材列表
    tags = df_preview[df_preview["id"] == recipe_id]["preview_tag"].tolist()

    # 詳細食材列表
    det = df_detailed[df_detailed["id"] == recipe_id]
    ingredients = det[["ingredient_name", "quantity", "unit"]].to_dict(orient="records")

    # 做法步驟
    st = df_steps[df_steps["id"] == recipe_id].sort_values("step_no")
    steps_list = st[["step_no", "description"]].to_dict(orient="records")

    # 組合結果
    rec_dict["preview_tags"] = tags
    rec_dict["ingredients"] = ingredients
    rec_dict["steps"] = steps_list

    return rec_dict


def search_similar(query: str, top_k: int = 5):
    """
    返回與查詢字串相似度最高的前 top_k 筆 (recipe_id, similarity_score)
    """
    q_emb = model.encode([query])[0]
    q_norm = np.linalg.norm(q_emb)
    sims = embeddings.dot(q_emb) / (emb_norms * q_norm + 1e-10)
    idxs = np.argsort(-sims)[:top_k]
    return [(int(tags[i]["id"]), float(sims[i])) for i in idxs]


def search_and_retrieve(query: str, top_k: int = 3):
    """
    查詢時僅回傳「所有關鍵字都出現」的食譜（ingredients/tag 必須全覆蓋），
    並根據語意分數排序，數量不足時不補部分命中，只顯示完全命中的結果。
    """
    # 1. 取得所有 tags（已在全域變數 tags 載入）
    from collections import defaultdict

    # 2. 將 tags 轉換為 id -> set(tag) 結構
    id2tags = defaultdict(set)
    for item in tags:
        id2tags[item["id"]].add(item["tag"])

    # 3. 將 query 拆成多個關鍵字
    # 支援中/英逗號
    keywords = [kw.strip() for kw in query.replace("，", ",").split(",") if kw.strip()]
    if not keywords:
        return []

    # 4. 找出同時擁有所有關鍵字的食譜 id
    full_hit_ids = [
        rid
        for rid, tagset in id2tags.items()
        if all(any(kw in tag for tag in tagset) for kw in keywords)
    ]

    if not full_hit_ids:
        return []

    # 5. 用 embedding 計算語意分數，只排序完全命中的 id
    q_emb = model.encode([query])[0]
    q_norm = np.linalg.norm(q_emb)
    sims = embeddings.dot(q_emb) / (emb_norms * q_norm + 1e-10)
    # 製作 id: max_score 字典
    id2score = {}
    for i, t in enumerate(tags):
        rid = int(t["id"])
        if rid in full_hit_ids:
            # 取這個 id 的最大語意分數（因為一個 id 可能對應多個 tag 向量）
            id2score[rid] = max(id2score.get(rid, float("-inf")), float(sims[i]))

    # 6. 按語意分數排序，取 top_k
    sorted_ids = sorted(full_hit_ids, key=lambda rid: -id2score[rid])[:top_k]

    # 7. 回傳完整食譜內容與分數
    results = []
    for rid in sorted_ids:
        recipe = get_recipe_by_id(rid, (df_cleaned, df_preview, df_detailed, df_steps))
        if recipe:
            results.append({"id": rid, "score": id2score[rid], "recipe": recipe})
    return results


def search_by_partial_ingredients(query, top_k=3):
    ingredients = [
        kw.strip() for kw in query.replace("，", ",").split(",") if kw.strip()
    ]
    if not ingredients:
        return []
    id2count = {}
    for rid, tagset in id2tags.items():
        count = sum(any(kw in tag for tag in tagset) for kw in ingredients)
        if count > 0:
            id2count[rid] = count  # 至少命中1個才納入
    if not id2count:
        return []
    q_emb = model.encode([query])[0]
    q_norm = np.linalg.norm(q_emb)
    sims = embeddings.dot(q_emb) / (emb_norms * q_norm + 1e-10)
    id2score = {}
    for i, t in enumerate(tags):
        rid = int(t["id"])
        if rid in id2count:
            id2score[rid] = max(id2score.get(rid, float("-inf")), float(sims[i]))
    sorted_ids = sorted(
        id2count.keys(), key=lambda rid: (-id2count[rid], -id2score[rid])
    )[:top_k]
    results = []
    for rid in sorted_ids:
        recipe = get_recipe_by_id(rid, (df_cleaned, df_preview, df_detailed, df_steps))
        if recipe:
            results.append(
                {
                    "id": rid,
                    "score": id2score[rid],
                    "matched_count": id2count[rid],
                    "recipe": recipe,
                }
            )
    return results


def call_ollama_llm(
    user_query: str, recipes: list, model: str = "qwen3:4b-q4_K_M"
) -> str:
    """
    給LLM明確指令：「請推薦最適合user_query的食譜，列出名稱、簡介、理由。」
    user_query：用戶輸入的食材/需求
    recipes：檢索到的完整食譜list（通常top3）
    """
    if not recipes:
        return "找不到符合的食譜。"

    # 組裝 context，每道菜名稱＋主要食材（只需重點資訊即可）
    context_blocks = []
    for r in recipes:
        rec = r["recipe"]
        # 只取重點內容：名稱、食材
        ingredients_str = "、".join(
            i["ingredient_name"] for i in rec.get("ingredients", [])
        )
        context_blocks.append(
            f"【{rec.get('食譜名稱','')}】(ID: {r['id']})\n"
            f"主要食材：{ingredients_str}\n"
            f"簡要說明：可參考詳細步驟製作。"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    # 推薦型 prompt，明確指令 LLM「請列出推薦名單、簡介與推薦理由」
    prompt = (
        f"以下是好幾道料理食譜的資訊：\n{context_text}\n\n"
        f"請扮演一位料理專家，根據這些食譜資訊，"
        f"推薦最適合『{user_query}』這個需求的料理，"
        f"請列出推薦食譜的名稱，簡單描述內容與推薦理由。\n"
        f"請在每道料理標題後標註其 ID（如：台式羅勒燒雞 (ID: 474705)），以便用戶後續查詢。\n"
        f"如果都很適合，可以用條列式分別推薦，每道請簡明說明為什麼推薦。\n"
        f"回覆請直接進入主題，不需討論分析過程。\n"
        f"只能從下列提供的食譜中推薦。\n"
        f"若發現內容重複，請合併為一條並只列一次。\n"
        f"請用繁體中文回答。"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Ollama 發生錯誤：{e.stderr.strip()}"


def pretty_print(item: dict):
    """以可讀文字格式列印單一道食譜結果，不含 preview_tags"""
    rec = item["recipe"]
    print(f"=== 查詢結果：Recipe ID {item['id']} (相似度 {item['score']:.4f}) ===\n")
    print(
        f"食譜名稱：{rec.get('食譜名稱','')}\n"
        f"分類　　　：{rec.get('category','')}\n"
        f"網址　　　：{rec.get('網址','')}\n"
        f"圖片路徑　：{rec.get('圖片相對路徑','')}\n"
    )
    # 食材
    print("── 食材 Ingredients ──")
    for idx, ing in enumerate(rec.get("ingredients", []), 1):
        print(
            f"{idx}. {ing.get('ingredient_name','')} {ing.get('quantity','')}{ing.get('unit','')}"
        )
    print()
    # 步驟
    print("── 步驟 Steps ──")
    for step in rec.get("steps", []):
        print(f"{step.get('step_no','')}. {step.get('description','')}")
    print()


if __name__ == "__main__":
    print("RAG智能推薦查詢（僅支援 OR 部分關鍵字查詢，輸入 exit 離開）")

    while True:
        query = input("\n請輸入查詢食材或需求: ").strip()
        if query.lower() in ("exit", "quit"):
            break

        # 只支援 OR 查詢
        res = search_by_partial_ingredients(query, top_k=3)

        if not res:
            print("\n找不到相關食譜。")

            choice = input("是否要使用瀏覽器搜尋？(y/n): ").strip().lower()
            if choice == "y":
                import webbrowser

                query_url = f"https://www.google.com/search?q={query}+食譜"
                webbrowser.open(query_url)
                print(f"🔍 已在瀏覽器中搜尋：「{query} 食譜」")
            else:
                print("您可以嘗試其他關鍵字。\n")

            continue

        print("\n正在自動推薦最適合的食譜...\n")
        answer = call_ollama_llm(query, res)
        print("🧠 智能推薦：\n" + answer + "\n")

        # 🔔 加入明確指引提示
        print(
            "🔍 若想查看其中一道食譜的【詳細食材與步驟】，請輸入該食譜『名稱關鍵字』或該食譜的 ID"
        )
        print("✏️ 若想重新查詢其他食材，請輸入 new；離開請輸入 exit。")

        # 構建名稱 → ID 對照
        name_map = {r["recipe"]["食譜名稱"]: r["id"] for r in res}
        id_set = set(r["id"] for r in res)

        while True:
            follow_up = input(
                "請輸入想查看詳情的食譜編號/名稱，或輸入 new 查詢新食材: "
            ).strip()
            if follow_up.lower() in ("exit", "quit"):
                exit()
            if follow_up.lower() in ("new", ""):
                break

            # 比對數字編號
            if follow_up.isdigit() and int(follow_up) in id_set:
                selected_id = int(follow_up)
            else:
                for name, rid in name_map.items():
                    if follow_up in name:
                        selected_id = rid
                        break

            if selected_id:
                recipe = get_recipe_by_id(
                    selected_id, (df_cleaned, df_preview, df_detailed, df_steps)
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
