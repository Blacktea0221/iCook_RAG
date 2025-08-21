# scripts/langchain_project/tools/recipe_search_tool.py
# -*- coding: utf-8 -*-
"""
LangChain Tool：食譜查詢（食材 OR 召回 → 重排）
- Input: user_text / top_k / candidate_ids
- Output: 結構化列表（id/title/preview_tags/score/link）
"""
from __future__ import annotations

from typing import List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from scripts.database.ingredient_utils import build_ingredient_set_from_db
from scripts.RAG.search_engine import pull_ingredients, tag_then_vector_rank


# ---------- Schemas ----------
class RecipeLite(BaseModel):
    id: int
    title: str
    preview_tags: List[str] = []
    score: float
    link: Optional[str] = None


class RecipeSearchInput(BaseModel):
    user_text: str = Field(..., description="使用者查詢句子，例如：我有九層塔和鮭魚")
    top_k: int = Field(5, ge=1, le=20, description="返回幾道食譜（重排後）")
    candidate_ids: List[int] = Field(
        default_factory=list, description="（可選）先驗過濾後的候選 ID"
    )


class RecipeSearchOutput(BaseModel):
    results: List[RecipeLite] = []
    recall_size: int = 0
    error: Optional[str] = None


# ---------- Tool ----------
class RecipeSearchTool(BaseTool):
    # Pydantic v2 需要型別註解
    name: str = "recipe_search_tool"
    description: str = "以食材/標籤做 OR 召回並重排，回傳精簡食譜列表。"
    args_schema: Type[RecipeSearchInput] = RecipeSearchInput

    def __init__(self):
        super().__init__()
        self._ingredient_set: Optional[set] = None

    def _ensure_ing(self) -> Optional[set]:
        """載入 DB 食材字典；若失敗或為空，回傳 None（代表之後不要過濾）。"""
        if self._ingredient_set is not None:
            return self._ingredient_set
        try:
            s = build_ingredient_set_from_db()
            if not s:
                print(
                    "[recipe_search_tool] ingredient_set is EMPTY → will NOT filter tokens"
                )
                self._ingredient_set = None
            else:
                print(f"[recipe_search_tool] ingredient_set loaded: {len(s)} terms")
                self._ingredient_set = s
        except Exception as e:
            print(f"[recipe_search_tool] build_ingredient_set_from_db() failed: {e}")
            self._ingredient_set = None
        return self._ingredient_set

    def _run(self, user_text: str, top_k: int, candidate_ids: List[int]) -> dict:
        try:
            # 1) 載入食材字典；若為 None/空集合，就不要做「在字典中的過濾」
            ing_set = self._ensure_ing()

            # 2) 用已灌好字典的 jieba 切詞；若無字典，讓 pull_ingredients 不過濾
            tokens = pull_ingredients(user_text, ing_set if ing_set else None)
            if not tokens:
                # 降級：粗略用標點/空白切一輪，避免 tokens 全空
                import re

                tokens = [t for t in re.split(r"[ ,，、。；;:：\t\n]+", user_text) if t]
            print(f"[recipe_search_tool] tokens={tokens[:10]} ...")

            # 3) 交給搜尋引擎（支援 candidate_ids 交集）
            raw = tag_then_vector_rank(
                user_text=user_text,
                tokens_from_jieba=tokens,
                top_k=top_k,
                candidate_ids=candidate_ids or None,
            )

            # 轉成穩定輸出
            items: List[RecipeLite] = []
            for r in raw:
                rec = r.get("recipe") or {}
                items.append(
                    RecipeLite(
                        id=int(r.get("id")),
                        title=rec.get("title") or "",
                        preview_tags=rec.get("preview_tags", []),
                        score=float(r.get("score", 0.0)),
                        link=rec.get("link"),
                    )
                )

            return RecipeSearchOutput(
                results=items,
                recall_size=len(raw),
                error=None,
            ).model_dump()

        except Exception as e:
            return RecipeSearchOutput(
                results=[], recall_size=0, error=str(e)
            ).model_dump()
