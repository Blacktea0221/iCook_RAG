# scripts/langchain_project/orchestrator_schemas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


# --- 第一層 Router ---
class IntentOut(BaseModel):
    intent: Literal["recipe", "nutrition", "price", "seasonal", "other"]


class ConstraintsSchema(BaseModel):
    """與 constraints_filter_tool 對齊的限制條件"""
    # 註：tool 目前接受的說明文字為 "vegetarian/omnivore/None"
    # 這裡用 Optional[str] 包起來，None / "none" 都視為無限制
    diet: Optional[str] = Field(default=None, description="vegetarian / omnivore / none")
    no_pork: bool = False
    extra_exclude: List[str] = Field(default_factory=list)


class RecipeRouteOut(BaseModel):
    """
    Sub-router 的結構化輸出：
    - by_ingredients：一般關鍵字/食材查詢
    - by_constraints：先過濾（diet/no_pork/關鍵字排除），之後再檢索
    - by_name：名稱/關鍵字直接全文檢索
    """
    sub_intent: Literal["by_ingredients", "by_constraints", "by_name"]
    constraints: Optional[ConstraintsSchema] = None
    name_query: Optional[str] = None
    # 預留這欄位，未來如果真的要把解析後的食材清單往下傳也能用
    ingredients: List[str] = Field(default_factory=list)


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
