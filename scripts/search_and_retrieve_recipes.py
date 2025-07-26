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
$ pip install pandas numpy sentence-transformers googlesearch-python jieba

執行：
$ python search_and_retrieve_recipes.py
"""
import json
import os
import re
import subprocess
import textwrap
from collections import defaultdict
from typing import Dict, List

import jieba
import numpy as np
import pandas as pd
from googlesearch import search  # pip install googlesearch-python
from sentence_transformers import SentenceTransformer

# -------------------- 專案路徑 --------------------
category = "九層塔"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# -------------------- 檔案路徑 --------------------
tags_path = os.path.join(ROOT_DIR, "data", "embeddings", category, "tags.json")
embed_path = os.path.join(ROOT_DIR, "data", "embeddings", category, "embeddings.npy")
cleaned_path = os.path.join(
    ROOT_DIR, "data", "clean", category, f"{category}_recipes_cleaned.csv"
)
preview_path = os.path.join(
    ROOT_DIR, "data", "clean", category, f"{category}_preview_ingredients.csv"
)
detailed_path = os.path.join(
    ROOT_DIR, "data", "clean", category, f"{category}_detailed_ingredients.csv"
)
steps_path = os.path.join(
    ROOT_DIR, "data", "clean", category, f"{category}_recipe_steps.csv"
)

# -------------------- 載入向量與模型 --------------------
with open(tags_path, "r", encoding="utf-8") as f:
    tags = json.load(f)
embeddings = np.load(embed_path)
model = SentenceTransformer("BAAI/bge-m3")
emb_norms = np.linalg.norm(embeddings, axis=1)

# 建立 id -> set(tag)
id2tags = defaultdict(set)
for item in tags:
    id2tags[item["id"]].add(item["tag"])

# -------------------- 載入食譜資料 --------------------
print("載入清理後的食譜資料...")
df_cleaned = pd.read_csv(cleaned_path, sep=";", encoding="utf-8-sig")
df_cleaned.columns = df_cleaned.columns.str.strip()
df_preview = pd.read_csv(preview_path, encoding="utf-8-sig").rename(
    columns=lambda x: x.strip()
)
df_detailed = pd.read_csv(detailed_path, encoding="utf-8-sig").rename(
    columns=lambda x: x.strip()
)
df_steps = pd.read_csv(steps_path, encoding="utf-8-sig").rename(
    columns=lambda x: x.strip()
)

# ==============================================================
#  ★★★ 新增區塊 1：準備「食材字典」 ★★★
# ==============================================================


def build_ingredient_set(df_preview: pd.DataFrame, df_detailed: pd.DataFrame) -> set:
    """從 preview_tag 與 ingredient_name 兩欄組成去重後的食材集合"""
    tags_set = set()
    # preview_tag 以逗號分隔
    for line in df_preview["preview_tag"]:
        tags_set.update(t.strip() for t in str(line).split(",") if t.strip())
    # detailed_ingredients
    tags_set.update(df_detailed["ingredient_name"].astype(str).str.strip())
    # 去空、去純數字
    return {t for t in tags_set if t and not re.fullmatch(r"\d+", t)}


# ★ 在 build_ingredient_set 定義之後、main loop 之前加上 ↓↓↓
ING_SET: set[str] = build_ingredient_set(df_preview, df_detailed)

# 把所有食材加進 Jieba 自訂字典，讓斷詞能一次切出完整詞
for w in ING_SET:
    jieba.add_word(w)

print(f"食材字典大小：{len(ING_SET)}")

# ==============================================================
#  ★★★ 新增區塊 2：關鍵字抽取函式 ★★★
# ==============================================================

LLM_PROMPT = """你是食材抽取助手，只回 JSON 陣列。從句子中找出食材名稱（只要名稱），依序輸出：
---
{text}
---"""


def jieba_extract(text: str) -> List[str]:
    """用 Jieba 斷詞後過白名單"""
    clean = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    tokens = jieba.lcut(clean, cut_all=False)
    return [ing for ing in ING_SET if ing in text]


def llm_extract(text: str, model_name: str = "qwen3:4b-q4_K_M") -> List[str]:
    """呼叫 Ollama 模型抽取關鍵字（後援）"""
    prompt = LLM_PROMPT.format(text=text)
    res = subprocess.run(
        ["ollama", "run", model_name, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    try:
        items = json.loads(res)
    except json.JSONDecodeError:
        items = re.split(r"[，,]\s*", res)
    # 只留字典內詞
    return [i.strip() for i in items if i.strip() in ING_SET]


def pull_ingredients(user_text: str) -> List[str]:
    """先用 Jieba，比對不到再用 LLM；回傳食材清單"""
    words = jieba_extract(user_text)
    return words if words else llm_extract(user_text)


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
        f"以下是料理食譜的資訊：\n{context_text}\n\n"
        f"請扮演一位料理專家，根據這些食譜資訊，"
        f"用20字描述內容。\n"
        f"請在每道料理標題後標註其 ID（如：台式羅勒燒雞 (ID: 474705)），以便用戶後續查詢。\n"
        f"將所有食譜條列式分別描述內容。\n"
        f"回覆請直接進入主題，不需討論分析過程。\n"
        f"只能從下列提供的食譜中描述內容。\n"
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


def google_search_recipes(keyword: str, k: int = 5) -> List[Dict]:
    """
    後備 Google 搜尋：在使用者輸入的文字後面自動加上「食譜」二字，
    並取回前 k 筆結果。
    回傳格式：[{ 'title': ..., 'link': ..., 'snippet': ... }, ...]
    """
    query = f"{keyword} 食譜"
    results = []
    for item in search(query, advanced=True, num_results=k, lang="zh-tw"):
        results.append(
            {"title": item.title, "link": item.url, "snippet": item.description}
        )
    return results


def summarize_search_results(
    user_query: str, results: list, model: str = "qwen3:4b-q4_K_M"
) -> str:
    """把多筆搜尋結果交給 LLM，請它用繁體中文歸納回答"""
    blocks = []
    for r in results:
        blocks.append(f"【{r['title']}】\n{r['snippet']}\nLink: {r['link']}")
    context = "\n\n---\n\n".join(blocks)

    prompt = textwrap.dedent(
        f"""\
        你將獲得來自 Google 搜尋「{user_query} 食譜」的結果摘要（如下 %%% 所示），請依據**僅提供的資訊**產出條列式清單。

        ✅ 每筆輸出請嚴格遵循以下格式（用全形逗號分隔）：
        網頁標題，全形逗號，20 字左右的簡介，全形逗號，原始網址

        ⚠️ 請注意：
        1. **只能基於提供的資訊內容回答，不得推論或自行補充**
        2. 每則簡介**長度約為 20 字（18～22 字內）**
        3. 結果以條列清單形式呈現，每筆結果獨立一行
        4. 請全程使用**繁體中文**
        5. 網址請保持原樣，不可修改或省略

        %%%
        {context}
        %%%
        """
    )

    res = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return res.stdout.strip()


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
    print("RAG 智能推薦查詢（輸入任何中文描述；exit 離開）")

    while True:
        raw_input_text = input("\n請描述你有的食材或需求: ").strip()
        if raw_input_text.lower() in ("exit", "quit"):
            break

        # 1) 先抽取食材關鍵字
        keywords = pull_ingredients(raw_input_text)
        if not keywords:
            print("⚠️ 未偵測到任何可用食材，改為網路搜尋模式…")
            # 直接做 Google 後備
            web_hits = google_search_recipes(raw_input_text, k=5)
            if not web_hits:
                print("🚫 Google 無結果，請嘗試其他關鍵字。")
                continue
            summary = summarize_search_results(raw_input_text, web_hits)
            print("🌐 來自 Google 的推薦：\n" + summary + "\n")
            # open_choice = input("要在瀏覽器開啟第一筆結果嗎？(y/n): ").strip().lower()
            # if open_choice == "y":
            #     import webbrowser

            #     webbrowser.open(web_hits[0]["link"])
            continue

        # 2) 有抽到關鍵字，就用本地 OR 檢索
        query = ", ".join(keywords)
        res = search_by_partial_ingredients(query, top_k=3)

        # 3) 如果本地查無結果，再跑 Google 後備
        if not res:
            print("⚠️ 本地資料庫查無結果，嘗試網路搜尋…")
            web_hits = google_search_recipes(query, k=5)
            if not web_hits:
                print("🚫 Google 無結果，請嘗試其他關鍵字。")
                continue
            summary = summarize_search_results(query, web_hits)
            print("🌐 來自 Google 的推薦：\n" + summary + "\n")
            # open_choice = input("要在瀏覽器開啟第一筆結果嗎？(y/n): ").strip().lower()
            # if open_choice == "y":
            #     import webbrowser

            #     webbrowser.open(web_hits[0]["link"])
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
                "請輸入想查看詳情的食譜編號/名稱，或輸入 new 查詢新食材: "
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
                    selected_id,
                    (df_cleaned, df_preview, df_detailed, df_steps),
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
