# scripts/langchain_project/services/recipe_name_lookup.py
"""
提供「食譜名稱」查詢（不做向量版）。
優先順序：
1) 完全相等
2) ILIKE 前綴 / 子字串
3) pg_trgm 相似度（若可用），門檻預設 0.35

回傳：List[{"id": int, "recipe": str}]
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

DB_CONFIG = dict(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)


def _connect():
    return psycopg2.connect(**DB_CONFIG)


def _normalize_name(q: str) -> str:
    """去除常見的括號備註與多餘空格。"""
    import re

    q = (q or "").strip()
    q = re.sub(r"[（(].*?[)）]", "", q)  # 去掉括號內容
    q = re.sub(r"\s+", "", q)
    return q


def search_by_recipe_name(q: str, limit: int = 5) -> List[Dict[str, Any]]:
    qn = _normalize_name(q)
    if not qn:
        return []

    sql = """
    WITH c AS (
      SELECT id, recipe,
        CASE
          WHEN recipe = %(q)s THEN 3
          WHEN recipe ILIKE %(q)s || '%%' THEN 2
          WHEN recipe ILIKE '%%' || %(q)s || '%%' THEN 1
          ELSE 0
        END AS match_level
      FROM main_recipe
    )
    SELECT id, recipe
    FROM c
    WHERE match_level > 0
    ORDER BY match_level DESC
    LIMIT %(k)s;
    """

    rows: List[Dict[str, Any]] = []
    try:
        with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, {"q": qn, "k": limit})
            rows = list(cur.fetchall())
    except Exception:
        rows = []

    # 若前面沒取滿，再用 trigram 相似度補齊（若可用）
    if len(rows) < limit:
        remain = limit - len(rows)
        try:
            with _connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, recipe
                    FROM main_recipe
                    WHERE similarity(recipe, %(q)s) >= 0.35
                    ORDER BY similarity(recipe, %(q)s) DESC
                    LIMIT %(k)s;
                    """,
                    {"q": qn, "k": remain},
                )
                more = list(cur.fetchall())
                # 去重
                seen = {r["id"] for r in rows}
                rows.extend([r for r in more if r["id"] not in seen])
        except Exception:
            pass  # 目標庫沒有安裝 pg_trgm 或權限不足

    return rows[:limit]
