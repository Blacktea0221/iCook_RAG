# scripts/langchain_project/schemas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --- 第一層 Router ---
class IntentOut(BaseModel):
    intent: Literal["recipe", "nutrition", "price", "seasonal", "other"]


# --- 第二層（只在 intent=recipe 時使用）---
class Constraints(BaseModel):
    diet: Optional[str] = None
    no_pork: bool = False
    no_beef: bool = False
    no_seafood: bool = False
    extra_exclude: List[str] = []


class RecipeRouteOut(BaseModel):
    # ✅ 改 2：sub_intent 同時接受中英文，讓 LLM 回中文不會驗證失敗
    sub_intent: Literal[
        "by_ingredients",
        "by_constraints",
        "by_name",
        "食譜查詢",
        "特殊需求",
        "食譜名稱",
    ] = "by_ingredients"
    name_query: Optional[str] = None
    ingredients: List[str] = []
    constraints: Constraints = Constraints()
    reason: str = ""


# --- Presenter 輸出（回給 LINE Bot/前端）---
class PresentItem(BaseModel):
    id: int
    title: str
    score: float
    link: Optional[str] = None


class PresentOut(BaseModel):
    intent: str
    items: List[PresentItem] = []
    summary_text: Optional[str] = None
