# scripts/langchain_project/tools/get_recipe_tool.py
# -*- coding: utf-8 -*-
"""
LangChain Tool：依 ID 取完整食譜資料
- Input: 一批 recipe ids、是否要帶步驟
- Output: 結構化完整食譜（title / preview_tags / ingredients / steps / link）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from scripts.RAG.search_engine import get_recipe_by_id


# ---------- Schemas ----------
class GetRecipeInput(BaseModel):
    ids: List[int] = Field(..., description="食譜 ID 列表")
    with_steps: bool = Field(False, description="是否附上烹調步驟")


class RecipeFull(BaseModel):
    id: int
    title: str
    preview_tags: List[str] = []
    ingredients: List[Dict[str, Any]] = []
    steps: Optional[List[Dict[str, Any]]] = None
    link: Optional[str] = None
    metadata: Dict[str, Any] = {}


class GetRecipeOutput(BaseModel):
    recipes: List[RecipeFull] = []
    error: Optional[str] = None


# ---------- Tool ----------
class GetRecipeTool(BaseTool):
    # 加上型別註解，避免 Pydantic v2 報錯
    name: str = "get_recipe_tool"
    description: str = "依 ID 讀取完整食譜（title/preview_tags/ingredients/steps/link）"
    args_schema: Type[GetRecipeInput] = GetRecipeInput

    def _run(self, ids: List[int], with_steps: bool) -> dict:
        try:
            out: List[RecipeFull] = []
            for rid in ids:
                rec = get_recipe_by_id(int(rid), with_steps=with_steps)
                if not rec:
                    continue
                out.append(
                    RecipeFull(
                        id=int(rec["id"]),
                        title=rec.get("title", ""),
                        preview_tags=rec.get("preview_tags", []),
                        ingredients=rec.get("ingredients", []),
                        steps=rec.get("steps"),
                        link=rec.get("link"),
                        metadata={},
                    )
                )
            return GetRecipeOutput(recipes=out, error=None).model_dump()
        except Exception as e:
            return GetRecipeOutput(recipes=[], error=str(e)).model_dump()
