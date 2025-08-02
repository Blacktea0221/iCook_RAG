import re

import jieba
import numpy as np

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


# =============== 向量檢索 ===============


def search_similar(query: str, model, embeddings, emb_norms, tags, top_k: int = 5):
    """
    返回與查詢字串相似度最高的前 top_k 筆 (recipe_id, similarity_score)
    """
    q_emb = model.encode([query])[0]
    q_norm = np.linalg.norm(q_emb)
    sims = embeddings.dot(q_emb) / (emb_norms * q_norm + 1e-10)
    idxs = np.argsort(-sims)[:top_k]
    return [(int(tags[i]["id"]), float(sims[i])) for i in idxs]


def search_by_partial_ingredients(
    query,
    id2tags,
    model,
    embeddings,
    emb_norms,
    tags,
    df_cleaned,
    df_preview,
    df_detailed,
    df_steps,
    top_k=3,
    allowed_ids=None,
):
    """
    ingredients: 關鍵字以 , 或 ，分隔，可以是多個食材
    允許 allowed_ids 過濾（如：只看素食/不吃豬肉等）
    回傳結果list，每筆是 dict: {id, score, matched_count, recipe}
    """
    ingredients = [
        kw.strip() for kw in query.replace("，", ",").split(",") if kw.strip()
    ]
    if not ingredients:
        return []
    id2count = {}
    for rid, tagset in id2tags.items():
        if allowed_ids is not None and rid not in allowed_ids:
            continue
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
        if rid in id2count and (allowed_ids is None or rid in allowed_ids):
            id2score[rid] = max(id2score.get(rid, float("-inf")), float(sims[i]))
    sorted_ids = sorted(
        id2count.keys(), key=lambda rid: (-id2count[rid], -id2score[rid])
    )[:top_k]
    results = []
    for rid in sorted_ids:
        recipe = get_recipe_by_id(rid, df_cleaned, df_preview, df_detailed, df_steps)
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
