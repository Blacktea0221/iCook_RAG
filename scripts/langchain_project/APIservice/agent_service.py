# scripts/langchain_project/services/agent_service.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.langchain_project.model import get_chat_model
from scripts.RAG import search_engine
from scripts.database.ingredient_utils import build_ingredient_set_from_db

load_dotenv()

# ==== 載入 prompts ====
AGENTS_DIR = Path(__file__).resolve().parent.parent  # .../scripts/langchain_project
PROMPTS_DIR = AGENTS_DIR / "prompts"
ROUTER_SYS = "你是嚴格的分類器，只能輸出 JSON。"
ROUTER_PROMPT = (PROMPTS_DIR / "router_prompt.txt").read_text(encoding="utf-8")
PRESENTER_SYS = (PROMPTS_DIR / "presenter_prompt.txt").read_text(encoding="utf-8")

# ==== LLM (OpenAI) ====
llm_router = get_chat_model("router")       # 溫度 0，小回覆、分類穩定
llm_presenter = get_chat_model("presenter") # 溫度 0.3，摘要較自然

# ==== 結構化輸出型別 ====
class Route(TypedDict):
    intent: Literal["recipe", "nutrition", "price", "identify", "other"]

class KeywordsOut(TypedDict):
    keywords: List[str]


# ==== 內部工具 ====
def _classify_with_llm(text: str) -> str:
    """用 LLM 依 router_prompt 分類；失敗回 other。"""
    # 1) 先嘗試 structured output
    try:
        chain = llm_router.with_structured_output(Route)
        data = chain.invoke(
            [
                SystemMessage(content=ROUTER_SYS),
                HumanMessage(content=ROUTER_PROMPT.replace("{text}", text)),
            ]
        )
        if isinstance(data, dict) and "intent" in data:
            return data["intent"]
    except Exception:
        pass

    # 2) 後備：要求 JSON 並解析
    try:
        res = llm_router.invoke(
            [
                SystemMessage(content=ROUTER_SYS),
                HumanMessage(content=ROUTER_PROMPT.replace("{text}", text)),
            ]
        )
        data = json.loads(getattr(res, "content", "{}") or "{}")
        if isinstance(data, dict) and "intent" in data:
            return data["intent"]
    except Exception:
        pass

    return "other"


def _format_presenter_user_msg(user_text: str, recipes: List[Dict]) -> str:
    """把 RAG 結果整理成 Presenter 的 user message 文字。"""
    if not recipes:
        return "（目前沒有任何食譜結果）"
    blocks = []
    for r in recipes:
        rec = r.get("recipe", {})
        title = (rec.get("recipe") or rec.get("食譜名稱") or "").strip()
        ingredients_str = "、".join(
            (i.get("ingredient") or i.get("ingredient_name") or "").strip()
            for i in rec.get("ingredients", [])
            if (i.get("ingredient") or i.get("ingredient_name"))
        )
        blocks.append(
            f"【{title}】 (ID: {r['id']})\n"
            f"主要食材：{ingredients_str}\n"
            f"說明：可參考詳細步驟製作。"
        )
    context_text = "\n\n---\n\n".join(blocks)
    return (
        f"以下是使用者查詢「{user_text}」取得的食譜清單，"
        f"每筆用【標題】與 ID 表示：\n\n{context_text}\n\n"
        f"請依系統提示產生條列輸出。"
    )


def _present_summary_text(user_text: str, hits: List[Dict]) -> str:
    """呼叫 Presenter 產出條列文字摘要。"""
    user_msg = _format_presenter_user_msg(user_text, hits)
    res = llm_presenter.invoke(
        [SystemMessage(content=PRESENTER_SYS), HumanMessage(content=user_msg)]
    )
    return getattr(res, "content", str(res))


def _extract_keywords_zh_nouns(text: str) -> List[str]:
    """用 LLM 從輸入中抽 3 個繁體中文『名詞』關鍵字（去重、最重要優先）。"""
    sys_msg = (
        "你是關鍵字抽取器。任務：\n"
        "1) 從使用者輸入中抽取『繁體中文名詞』關鍵字。\n"
        "2) 僅輸出最重要的 3 個；若不足 3 個就輸出現有的。\n"
        "3) 僅能用輸入中出現過的詞；需去重、避免同義重複。\n"
        "4) 嚴格輸出 JSON，格式：{\"keywords\": [\"詞1\",\"詞2\",\"詞3\"]}。\n"
    )
    try:
        chain = llm_router.with_structured_output(KeywordsOut)
        data = chain.invoke([SystemMessage(content=sys_msg), HumanMessage(content=text)])
        kws = data.get("keywords", []) if isinstance(data, dict) else []
        # 保底：最多 3 個、去空白
        kws = [k.strip() for k in kws if k and k.strip()]
        return kws[:3]
    except Exception:
        # 後備：嘗試一般 JSON 解析
        res = llm_router.invoke([SystemMessage(content=sys_msg), HumanMessage(content=text)])
        try:
            data = json.loads(getattr(res, "content", "{}") or "{}")
            kws = data.get("keywords", [])
            kws = [k.strip() for k in kws if k and k.strip()]
            return kws[:3]
        except Exception:
            return []


# ==== 對外：API 服務入口 ====
def route_and_struct(text: str, top_k: int = 5) -> Dict:
    """
    回傳統一結構：
      - 食譜：{"intent":"recipe","payload":{"summary_text":"...","hits":[...]}}
      - 其他：{"intent":"other|nutrition|price|identify","payload":{"keywords":[...]}}
    """
    intent = _classify_with_llm(text)

    if intent == "recipe":
        # 1) RAG 查詢（沿用你的 search_engine）
        ing_set = build_ingredient_set_from_db()
        tokens = search_engine.pull_ingredients(text, ing_set)
        hits = search_engine.tag_then_vector_rank(text, tokens_from_jieba=tokens, top_k=top_k)

        # 2) Presenter 產出文字摘要（即使有 hits 也會同時產出）
        summary_text = _present_summary_text(text, hits) if hits else "找不到符合的食譜。"

        return {"intent": "recipe", "payload": {"summary_text": summary_text, "hits": hits}}

    # 非食譜：抽 3 個繁中名詞當關鍵字
    keywords = _extract_keywords_zh_nouns(text)
    if not keywords:
        keywords = []
    return {"intent": intent, "payload": {"keywords": keywords}}
