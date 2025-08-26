# scripts/RAG/search_engine.py
# -*- coding: utf-8 -*-
"""
搜尋引擎核心 API（被 LangChain Tools 呼叫）
------------------------------------------------
職責：封裝資料庫查詢、OR 召回、（可選的）向量重排、以及食譜資料讀取。
**不要**在這裡做 LLM 摘要與呈現；上層 Orchestrator/Presenter 處理即可。

你可以先用「簡易重排（標籤重合度）」跑通流程；之後有 pgvector 時，把
`_rerank_by_vector` 的內容替換即可（其餘程式不用改）。
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
import psycopg2.extras

# =========
# 連線 & DB helpers
# =========


def _get_dsn() -> str:
    """
    從環境變數讀取 PostgreSQL 連線資訊：
    - PG_DSN（優先）或 PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD
    """
    if os.getenv("PG_DSN"):
        return os.getenv("PG_DSN")  # e.g. postgresql://user:pass@host:5432/db
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = int(os.getenv("PG_PORT", "5432"))
    db = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    pwd = os.getenv("PG_PASSWORD", "")
    return f"host={host} port={port} dbname={db} user={user} password={pwd}"


def _get_conn():
    """取得 psycopg2 連線（每次用完即關閉；避免全域連線佔用）"""
    return psycopg2.connect(_get_dsn())


def fetch_all(sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    """
    執行查詢並回傳 List[dict]，列名作為鍵。
    方便 tools 與上層使用。
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def fetch_one(
    sql: str, params: Optional[Sequence[Any]] = None
) -> Optional[Dict[str, Any]]:
    """回傳單筆 dict 或 None。"""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    return dict(row) if row else None


# =========
# 食材抽取（由上層提供 ingredient_set 時效果最佳）
# =========


def pull_ingredients(user_text: str, ingredient_set: Optional[set] = None) -> List[str]:
    """
    使用者句子 → 食材 tokens
    - 若提供 ingredient_set，會以「在字典中」做濾除，降低雜訊。
    - 你已在 ingredient_utils 中把資料庫的食材灌進 jieba 字典。
    """
    try:
        import jieba  # 本地斷詞
    except Exception:
        return []

    tokens = [t.strip() for t in jieba.lcut(user_text) if t.strip()]
    if ingredient_set:
        tokens = [t for t in tokens if t in ingredient_set]
    # 去重、保序
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# =========
# OR 召回（ingredient / preview_tag）
# =========


def _candidate_recipe_ids_by_tag(tags: List[str], limit: int = 200) -> List[int]:
    """
    根據 tags（ingredient 或 preview_tag）做 OR 召回，回傳 recipe_id 列表。
    - 以命中數量排序（命中越多越前面）
    - 你可以視需要加上 AND 邏輯（例如最低命中2個再納入）
    """
    if not tags:
        return []
    sql = """
        SELECT recipe_id, COUNT(*) AS hit
        FROM public.ingredient
        WHERE ingredient = ANY(%s) OR preview_tag = ANY(%s)
        GROUP BY recipe_id
        ORDER BY hit DESC
        LIMIT %s
    """
    rows = fetch_all(sql, (tags, tags, limit))
    return [int(r["recipe_id"]) for r in rows]


# =========
# 重排（預設：標籤重合度；之後可換向量重排/pgvector）
# =========


def _overlap_score(query_tokens: List[str], recipe_tags: Iterable[str]) -> float:
    """簡易評分：|交集| / sqrt(|Q| * |Doc|)，避免長度偏差。"""
    q = set(query_tokens)
    d = set([t for t in recipe_tags if t])
    if not q or not d:
        return 0.0
    inter = len(q & d)
    denom = math.sqrt(len(q) * len(d))
    return inter / denom if denom > 0 else 0.0


def _fetch_recipe_tags_for(ids: List[int]) -> Dict[int, List[str]]:
    if not ids:
        return {}
    sql = """
        SELECT recipe_id, COALESCE(preview_tag, ingredient) AS tag
        FROM public.ingredient
        WHERE recipe_id = ANY(%s)
    """
    rows = fetch_all(sql, (ids,))
    bag: Dict[int, List[str]] = {}
    for r in rows:
        rid = int(r["recipe_id"])
        raw = (r.get("tag") or "").strip()
        # 原字串也保留
        parts = [raw] if raw else []
        # 額外：用常見分隔符切成單詞一併加入
        parts += [t.strip() for t in re.split(r"[ ,，、/|｜]+", raw) if t.strip()]
        bag.setdefault(rid, []).extend(parts)
    return bag


def _rerank_by_overlap(
    candidate_ids: List[int], query_tokens: List[str], top_k: int
) -> List[Tuple[int, float]]:
    """
    預設重排：以標籤/食材重合度做分數。
    想切到「向量重排」：只要把這個函式改掉（或呼叫 pgvector 相似度），其他地方不用動。
    """
    tag_bag = _fetch_recipe_tags_for(candidate_ids)
    scored: List[Tuple[int, float]] = []
    for rid in candidate_ids:
        tags = tag_bag.get(rid, [])
        score = _overlap_score(query_tokens, tags)
        scored.append((rid, float(score)))
    # 由高到低排序
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# =========
# 封裝的主流程（給 tool 用）
# =========


def tag_then_vector_rank(
    user_text: str,
    tokens_from_jieba: Optional[List[str]] = None,
    top_k: int = 5,
    candidate_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    1) 展開/清洗 tokens（若沒給就用 pull_ingredients 簡單切）
    2) OR 召回（ingredient/preview_tag）
       - 若有 candidate_ids（例如先經過 constraints 過濾），會與召回集取 **交集**
    3) 重排（預設重合度；未來可換向量重排）
    4) 回傳前 top_k 的 {id, score, recipe(精簡欄位)}
    """
    tokens = tokens_from_jieba or pull_ingredients(user_text, None)
    if not tokens:
        # fallback：用簡單分隔符再試一輪
        rough = [t for t in re.split(r"[ ,，、。]+", user_text) if t]
        tokens = rough

    # 召回
    recall_ids = _candidate_recipe_ids_by_tag(tokens, limit=200)
    if candidate_ids:
        base = set(int(x) for x in candidate_ids)
        recall_ids = [rid for rid in recall_ids if rid in base]
    if not recall_ids:
        return []

    # 重排（預設重合度）
    ranked = _rerank_by_overlap(recall_ids, tokens, top_k=top_k)
    top_ids = [rid for rid, _ in ranked]
    id2score = dict(ranked)

    # 取精簡欄位
    lite = fetch_lite(top_ids)
    out: List[Dict[str, Any]] = []
    for it in lite:
        rid = int(it["id"])
        out.append(
            {
                "id": rid,
                "score": float(id2score.get(rid, 0.0)),
                "recipe": {
                    "title": it.get("title", "") or it.get("recipe", ""),
                    "preview_tags": it.get("preview_tags", []),
                    "link": it.get("link"),
                },
            }
        )
    # 依分數再排一次（避免 DB 回傳順序影響）
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


# =========
# 取資料（lite / full）
# =========


def fetch_lite(ids: List[int]) -> List[Dict[str, Any]]:
    """
    取列表顯示需要的欄位：
    - title（main_recipe.recipe 或 main_recipe.title）
    - preview_tags（聚合）
    - link（若有欄位可放）
    """
    if not ids:
        return []
    sql_title = """
        SELECT id, COALESCE(recipe, title) AS title
        FROM public.main_recipe
        WHERE id = ANY(%s)
    """
    titles = {int(r["id"]): r["title"] for r in fetch_all(sql_title, (ids,))}
    # 聚合 preview_tag
    sql_tags = """
        SELECT recipe_id, ARRAY_REMOVE(ARRAY_AGG(DISTINCT preview_tag), NULL) AS preview_tags
        FROM public.ingredient
        WHERE recipe_id = ANY(%s)
        GROUP BY recipe_id
    """
    tag_rows = fetch_all(sql_tags, (ids,))
    id2tags = {int(r["recipe_id"]): list(r["preview_tags"] or []) for r in tag_rows}

    # 若有網址欄位可在這裡補，例如 main_recipe.link
    sql_link = """
        SELECT id, NULL::text AS link
        FROM public.main_recipe
        WHERE id = ANY(%s)
    """
    link_rows = fetch_all(sql_link, (ids,))
    id2link = {int(r["id"]): r.get("link") for r in link_rows}

    out = []
    for rid in ids:
        out.append(
            {
                "id": int(rid),
                "title": titles.get(int(rid), ""),
                "preview_tags": id2tags.get(int(rid), []),
                "link": id2link.get(int(rid)),
            }
        )
    return out


def get_recipe_by_id(
    recipe_id: int, with_steps: bool = False
) -> Optional[Dict[str, Any]]:
    """
    讀取完整食譜（基本欄位 + 食材 + 可選步驟）
    """
    base = fetch_one(
        """
        SELECT id, COALESCE(recipe, title) AS title
        FROM public.main_recipe
        WHERE id = %s
        """,
        (recipe_id,),
    )
    if not base:
        return None

    ings = fetch_all(
        """
        SELECT ingredient, preview_tag
        FROM public.ingredient
        WHERE recipe_id = %s
        """,
        (recipe_id,),
    )

    steps: Optional[List[Dict[str, Any]]] = None
    if with_steps:
        rows = fetch_all(
            """
            SELECT step_no, step_desc
            FROM public.recipe_steps
            WHERE recipe_id = %s
            ORDER BY step_no ASC
            """,
            (recipe_id,),
        )
        steps = [{"no": int(r["step_no"]), "desc": r["step_desc"]} for r in rows]

    return {
        "id": int(base["id"]),
        "title": base["title"],
        "ingredients": [
            {"ingredient": r.get("ingredient"), "preview_tag": r.get("preview_tag")}
            for r in ings
        ],
        "steps": steps,
        "link": None,  # 若主表有網址欄位可放上來
        "preview_tags": list(
            {(r.get("preview_tag") or "").strip() for r in ings if r.get("preview_tag")}
        ),
    }
