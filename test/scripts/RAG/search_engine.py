import re

import jieba

from .vectorstore_utils import search_vectorstore

# =============== 關鍵字抽取相關 ===============


def jieba_extract(text: str, ingredient_set: set) -> list:
    """
    用 Jieba 斷詞後，過 ingredient_set 白名單，只回傳食材名稱
    """
    clean = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    tokens = jieba.lcut(clean, cut_all=False)
    return [ing for ing in ingredient_set if ing in text]


def pull_ingredients(user_text: str, ingredient_set: set) -> list:
    """
    關鍵字抽取，預設只用 Jieba（不自動 fallback LLM）
    """
    words = jieba_extract(user_text, ingredient_set)
    return words


# =============== 食譜ID查詢 ===============


def get_recipe_by_id(recipe_id, df_cleaned, df_preview, df_detailed, df_steps):
    """
    根據 recipe_id 從已載入的 DataFrame 中取得完整食譜資料。
    返回一個 dict 包括：
      - 主表欄位
      - preview_tags（list）
      - ingredients（list of dict）
      - steps（list of dict）
    """
    rec = df_cleaned[df_cleaned["id"] == recipe_id]
    if rec.empty:
        return None
    rec_dict = rec.iloc[0].to_dict()

    tags = df_preview[df_preview["id"] == recipe_id]["preview_tag"].tolist()
    det = df_detailed[df_detailed["id"] == recipe_id]
    ingredients = det[["ingredient_name", "quantity", "unit"]].to_dict(orient="records")
    st = df_steps[df_steps["id"] == recipe_id].sort_values("step_no")
    steps_list = st[["step_no", "description"]].to_dict(orient="records")

    rec_dict["preview_tags"] = tags
    rec_dict["ingredients"] = ingredients
    rec_dict["steps"] = steps_list
    return rec_dict


# =============== 向量檢索（全新） ===============


def search_similar(query: str, top_k: int = 5):
    """
    直接用 vectorstore_utils 查詢相似tag，回傳結果list: [{id, tag, vege_name, score, text}]
    """
    results = search_vectorstore(query, top_k=top_k)
    return results


def search_by_partial_ingredients(
    query,
    df_cleaned,
    df_preview,
    df_detailed,
    df_steps,
    top_k=3,
    allowed_ids=None,
):
    """
    ingredients: 關鍵字以 , 或 ，分隔，可以是多個食材
    allowed_ids: 用來進行條件過濾（如只看素食/不吃豬肉等）
    回傳結果list，每筆是 dict: {id, tag, vege_name, recipe}
    """
    # 關鍵字組成查詢句
    ingredients = [
        kw.strip() for kw in query.replace("，", ",").split(",") if kw.strip()
    ]
    if not ingredients:
        return []
    search_phrase = " ".join(ingredients)
    candidates = search_vectorstore(
        search_phrase, top_k=top_k * 3
    )  # 多抓一些做二次篩選

    results = []
    seen_ids = set()
    for cand in candidates:
        rid = int(cand["id"])
        if allowed_ids is not None and rid not in allowed_ids:
            continue
        if rid in seen_ids:
            continue
        recipe = get_recipe_by_id(rid, df_cleaned, df_preview, df_detailed, df_steps)
        if recipe:
            results.append(
                {
                    "id": rid,
                    "tag": cand["tag"],
                    "vege_name": cand["vege_name"],
                    "score": cand.get("score"),
                    "recipe": recipe,
                }
            )
            seen_ids.add(rid)
        if len(results) >= top_k:
            break
    return results
