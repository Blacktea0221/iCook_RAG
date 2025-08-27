# scripts/langchain_project/orchestrator_schemas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --- 第一層 Router ---
class IntentOut(BaseModel):
    intent: Literal["recipe", "nutrition", "price", "seasonal", "other"]


# 防止編輯器誤報，等同四選一+other
# 上行寫法只是保守處理，有需要可改回：
# intent: Literal["recipe", "nutrition", "price", "seasonal", "other"]


class ConstraintsSchema(BaseModel):
    diet: Optional[str] = Field(
        default=None, description="vegetarian / omnivore / none"
    )
    no_pork: bool = False
    extra_exclude: List[str] = Field(default_factory=list)


class RecipeRouteOut(BaseModel):
    sub_intent: Literal["by_ingredients", "by_constraints", "by_name"]
    constraints: Optional[ConstraintsSchema] = None
    name_query: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)


# --- Presenter 輸出 ---
class PresentItem(BaseModel):
    id: int
    title: str
    score: float
    link: Optional[str] = None  # ← 新增


class PresentOut(BaseModel):
    intent: str
    items: List[PresentItem] = []
    summary_text: Optional[str] = None
