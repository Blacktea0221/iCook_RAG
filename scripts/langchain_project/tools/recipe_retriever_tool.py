# scripts/langchain_project/tools/recipe_retriever.py
# -*- coding: utf-8 -*-
"""
LangChain Tool：全文檢索（pgvector on main_recipe.embedding）
- Input: query_text / top_k / candidate_ids(optional)
- Output: 與 RecipeSearchTool 相同結構（id/title/preview_tags/score/link）
"""
from __future__ import annotations

from typing import List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from scripts.RAG.search_engine import fetch_all, fetch_lite  # DB helpers
from scripts.RAG.vectorstore_utils import embed_text_to_np  # 你的既有 embedding 函式


# 為了與現有 Presenter/Orchestrator 對齊，沿用 RecipeLite 結構
class RecipeLite(BaseModel):
    id: int
    title: str
    preview_tags: List[str] = []
    score: float
    link: Optional[str] = None


class RecipeRetrieverInput(BaseModel):
    query_text: str = Field(..., description="任意查詢文字（如：三杯雞、蛋炒飯）")
    top_k: int = Field(5, ge=1, le=20, description="返回幾道食譜")
    candidate_ids: List[int] = Field(
        default_factory=list, description="（可選）外層先驗過濾後的候選 ID"
    )


class RecipeRetrieverOutput(BaseModel):
    results: List[RecipeLite] = []
    recall_size: int = 0
    error: Optional[str] = None


class RecipeRetrieverTool(BaseTool):
    name: str = "recipe_retriever_tool"
    description: str = (
        "以 pgvector 對 main_recipe.embedding 做全文相似檢索，回傳精簡食譜列表。"
    )
    args_schema: Type[RecipeRetrieverInput] = RecipeRetrieverInput

    def _run(self, query_text: str, top_k: int, candidate_ids: List[int]) -> dict:
        try:
            # 1) 文字 → 向量
            qvec = embed_text_to_np(query_text)
            if qvec is None or getattr(qvec, "size", 0) == 0:
                return RecipeRetrieverOutput(
                    results=[], recall_size=0, error="embedding 失敗"
                ).model_dump()

            # 2) 以 pgvector 距離做初選（取較大 buffer）
            base_limit = max(100, top_k * 20)
            embedding_str = str(qvec.tolist()).replace(" ", "")

            where_sql = ""
            params: List[object] = []
            if candidate_ids:
                where_sql = "WHERE id = ANY(%s)"
                params.append(candidate_ids)

            sql = f"""
                SELECT id,
                       COALESCE(recipe, title) AS name,
                       embedding <-> '{embedding_str}' AS distance
                FROM public.main_recipe
                {where_sql}
                ORDER BY distance ASC
                LIMIT {base_limit};
            """
            rows = fetch_all(
                sql, tuple(params) if params else None
            )  # :contentReference[oaicite:2]{index=2}
            if not rows:
                return RecipeRetrieverOutput(
                    results=[], recall_size=0, error=None
                ).model_dump()

            # 3) 只取前 top_k，並補最小欄位（title/preview_tags/link）
            rows = rows[:top_k]
            top_ids = [int(r["id"]) for r in rows]
            lite_rows = fetch_lite(
                top_ids
            )  # title/preview_tags/link :contentReference[oaicite:3]{index=3}
            id2lite = {int(r["id"]): r for r in lite_rows}

            items: List[RecipeLite] = []
            for r in rows:
                rid = int(r["id"])
                dist = float(r.get("distance", 0.0))
                # 以 1/(1+distance) 做直觀分數映射（距離越小分數越高）
                score = 1.0 / (1.0 + dist)
                meta = id2lite.get(rid, {})
                items.append(
                    RecipeLite(
                        id=rid,
                        title=meta.get("title", r.get("name", "")) or "",
                        preview_tags=meta.get("preview_tags", []) or [],
                        score=float(score),
                        link=meta.get("link"),
                    )
                )

            return RecipeRetrieverOutput(
                results=items, recall_size=len(items), error=None
            ).model_dump()

        except Exception as e:
            return RecipeRetrieverOutput(
                results=[], recall_size=0, error=str(e)
            ).model_dump()
