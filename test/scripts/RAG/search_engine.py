import re

import jieba
import psycopg2

from .vectorstore_utils import search_vectorstore

DB_CONFIG = dict(
    host="localhost", port=5432, database="postgres", user="lorraine", password="0000"
)


def fetch_one(query, params=None):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query, params or ())
    row = cur.fetchone()
    colnames = [desc[0] for desc in cur.description]
    conn.close()
    return dict(zip(colnames, row)) if row else None


def fetch_all(query, params=None):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query, params or ())
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    conn.close()
    return [dict(zip(colnames, row)) for row in rows]


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
def get_recipe_by_id(recipe_id):
    # 主表
    recipe = fetch_one(
        "SELECT * FROM public.recipes_cleaned WHERE id = %s;", (recipe_id,)
    )
    if not recipe:
        return None

    # 預覽tag
    tags = fetch_all(
        "SELECT preview_tag FROM public.preview_ingredients WHERE id = %s;",
        (recipe_id,),
    )
    recipe["preview_tags"] = [t["preview_tag"] for t in tags]

    # 食材
    ingredients = fetch_all(
        "SELECT ingredient_name, quantity, unit FROM public.detailed_ingredients WHERE id = %s;",
        (recipe_id,),
    )
    recipe["ingredients"] = ingredients

    # 步驟
    steps = fetch_all(
        "SELECT step_no, description FROM public.recipe_steps WHERE id = %s ORDER BY step_no;",
        (recipe_id,),
    )
    recipe["steps"] = steps

    return recipe


# 保留原有的向量查詢流程，僅在取細節時呼叫 get_recipe_by_id
from .vectorstore_utils import search_vectorstore

# =============== 向量檢索（全新） ===============


def search_similar(query: str, top_k: int = 5):
    """
    直接用 vectorstore_utils 查詢相似tag，回傳結果list: [{id, tag, vege_name, score, text}]
    """
    results = search_vectorstore(query, top_k=top_k)
    return results


def search_by_partial_ingredients(query, top_k=3, allowed_ids=None):
    """
    query: 關鍵字組成的查詢句（如：九層塔, 雞肉）
    allowed_ids: 可以限定 id（如素食、排除豬肉），可選
    返回 [{id, tag, vege_name, score, recipe}]
    """

    # ingredients 關鍵字組查詢
    ingredients = [
        kw.strip() for kw in query.replace("，", ",").split(",") if kw.strip()
    ]
    if not ingredients:
        return []

    search_phrase = " ".join(ingredients)
    candidates = search_vectorstore(search_phrase, top_k=top_k * 3)

    results = []
    seen_ids = set()
    for cand in candidates:
        rid = int(cand["id"])
        if allowed_ids is not None and rid not in allowed_ids:
            continue
        if rid in seen_ids:
            continue
        recipe = get_recipe_by_id(rid)
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
