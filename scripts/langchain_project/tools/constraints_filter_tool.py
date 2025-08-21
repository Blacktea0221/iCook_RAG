# scripts/langchain_project/tools/constraints_filter_tool.py
# -*- coding: utf-8 -*-
"""
LangChain Tool：特殊需求過濾（Constraints Filter）
- 依 diet / no_pork / 關鍵字排除，回傳一批 recipe_id
- 可選 base_ids 交集
"""
from __future__ import annotations

from typing import List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from scripts.RAG.search_engine import fetch_all  # DB helper


# ---------- Schemas ----------
class ConstraintsInput(BaseModel):
    base_ids: List[int] = Field(default_factory=list)
    diet: Optional[str] = Field(default=None, description="vegetarian/omnivore/None")
    no_pork: bool = False
    extra_exclude: List[str] = Field(default_factory=list)
    top_k: int = Field(50, ge=1, le=500)


class ConstraintsOutput(BaseModel):
    ids: List[int] = []
    reason: str = ""
    error: Optional[str] = None


# ---------- Tool ----------
class ConstraintsFilterTool(BaseTool):
    # 重要：這三個屬性一定要有「型別註解」，避免 Pydantic v2 報錯
    name: str = "constraints_filter_tool"
    description: str = (
        "依 diet/no_pork/關鍵字排除 過濾 recipe_id；可在 base_ids 範圍內做交集"
    )
    args_schema: Type[ConstraintsInput] = ConstraintsInput

    def _run(
        self,
        base_ids: List[int],
        diet: Optional[str],
        no_pork: bool,
        extra_exclude: List[str],
        top_k: int,
    ) -> dict:
        try:
            where, params = [], []

            if diet == "vegetarian":
                where.append("dg.vegetarian = TRUE")
            elif diet == "omnivore":
                # 不限制葷素（保留擴充空間）
                pass

            if no_pork:
                where.append("(COALESCE(dg.uses_pork, FALSE) = FALSE)")

            if base_ids:
                where.append("dg.recipe_id = ANY(%s)")
                params.append(base_ids)

            exclude_sql = ""
            if extra_exclude:
                exclude_sql = """
                    AND dg.recipe_id NOT IN (
                        SELECT DISTINCT i.recipe_id
                        FROM public.ingredient AS i
                        WHERE i.ingredient = ANY(%s) OR i.preview_tag = ANY(%s)
                    )
                """
                params.extend([extra_exclude, extra_exclude])

            sql = f"""
                SELECT dg.recipe_id
                FROM public.dietary_groups AS dg
                {"WHERE " + " AND ".join(where) if where else "WHERE TRUE"}
                {exclude_sql}
                LIMIT %s
            """
            params.append(top_k)

            rows = fetch_all(sql, tuple(params))
            ids = [int(r["recipe_id"]) for r in rows]

            reasons = []
            if diet:
                reasons.append(f"diet={diet}")
            if no_pork:
                reasons.append("no_pork")
            if extra_exclude:
                reasons.append(f"exclude({len(extra_exclude)})")
            if base_ids:
                reasons.append(f"intersect_base({len(base_ids)})")

            return ConstraintsOutput(
                ids=ids,
                reason=", ".join(reasons) or "no constraints",
                error=None,
            ).model_dump()
        except Exception as e:
            return ConstraintsOutput(ids=[], reason="", error=str(e)).model_dump()
