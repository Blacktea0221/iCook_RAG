# scripts/langchain_project/agents/recipe_agent.py
"""
CLI 版本的兩層鏈：
- 第一層 Router：判斷 intent
- 第二層（當 intent=recipe）：子路由到「食譜查詢 / 特殊需求 / 食譜名稱」
"""
from __future__ import annotations

import json
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.database.ingredient_utils import build_ingredient_set_from_db
from scripts.langchain_project.APIservice.agent_service import (
    _classify_intent,
    _classify_sub_intent,
    _present_summary_text,
)
from scripts.langchain_project.model import get_chat_model
from scripts.langchain_project.services.recipe_name_lookup import search_by_recipe_name
from scripts.RAG import search_engine

load_dotenv()


def build_agent():
    router_llm = get_chat_model("router")
    sub_router_llm = get_chat_model("sub_router")
    presenter_llm = get_chat_model("presenter")

    def run(user_text: str, top_k: int = 5) -> str:
        intent = _classify_intent(user_text)
        if intent != "recipe":
            return f"[{intent}]：{user_text}"

        sub = _classify_sub_intent(user_text)
        if sub["sub_intent"] == "食譜名稱":
            q = (sub.get("name_query") or "").strip() or user_text.strip()
            rows = search_by_recipe_name(q, limit=top_k)
            hits: List[Dict] = []
            for r in rows:
                full = search_engine.get_recipe_by_id(r["id"])
                if full:
                    hits.append(
                        {
                            "id": int(r["id"]),
                            "tag": None,
                            "vege_name": full.get("vege_name"),
                            "score": 0.0,
                            "recipe": full,
                        }
                    )
            if hits:
                return _present_summary_text(user_text, hits)
            # fallback
            sub["sub_intent"] = "食譜查詢"

        if sub["sub_intent"] == "特殊需求":
            cons = sub.get("constraints") or {}
            return f"[特殊需求] 偵測到約束：{json.dumps(cons, ensure_ascii=False)}（CLI 範例僅輸出，API 會交由你同伴服務處理）"

        # 食譜查詢
        ing_set = build_ingredient_set_from_db()
        tokens = search_engine.pull_ingredients(user_text, ing_set)
        hits = search_engine.tag_then_vector_rank(
            user_text, tokens_from_jieba=tokens, top_k=top_k
        )
        if not hits:
            return "找不到符合的食譜。"
        return _present_summary_text(user_text, hits)

    return run
