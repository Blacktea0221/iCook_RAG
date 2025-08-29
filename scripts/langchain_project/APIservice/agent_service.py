# scripts/langchain_project/APIservice/agent_service.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import jieba
from braintrust import init_logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BT_PROJECT = os.getenv("BT_PROJECT", "iCook-RAG")
bt_logger = init_logger(project=BT_PROJECT)

from scripts.langchain_project.orchestrator_graph import run_orchestrator

# 如專案已有 WebSearchTool，保留以下 import；若檔名/路徑不同請按實際調整
try:
    from scripts.langchain_project.tools.web_search_tool import (
        WebSearchInput,
        WebSearchTool,
    )
except Exception:
    WebSearchInput = None
    WebSearchTool = None

# 用 DB 建立的食材詞典，供 jieba 抽詞用
from scripts.database.ingredient_utils import build_ingredient_set_from_db

app = FastAPI(
    title="iCook RAG Orchestrator (Compat Output)", docs_url="/docs", redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def home():
    return {"ok": True, "msg": "iCook RAG API is alive."}


@app.get("/healthz", include_in_schema=False)
def healthz():
    return "ok"


# ---- Request/Response Schemas（固定回傳舊格式：intent + payload） ----------------


class RouteRequest(BaseModel):
    # Orchestrator 路徑（一般使用）
    text: Optional[str] = Field(
        default=None, description="使用者輸入文字（走 orchestrator）"
    )
    top_k: int = Field(default=5, ge=1, le=10)

    # 僅網路搜尋：傳 mode=web_search_only + prompt
    mode: Optional[str] = Field(
        default=None, description="若為 web_search_only，則走網路搜尋路徑"
    )
    prompt: Optional[str] = Field(
        default=None, description="web_search_only 模式下的查詢字串"
    )


class RouteResponse(BaseModel):
    intent: str
    payload: Dict[str, Any]


# ---- Utilities --------------------------------------------------------------


def _normalize_intent_to_output(orchestrator_intent: str) -> str:
    """把 orchestrator 的 intent 轉成對 LINE Bot 約定的 intent 文案。"""
    if orchestrator_intent == "seasonal":
        return "當季蔬菜月份"
    return orchestrator_intent


def _extract_keywords_via_db_jieba(text: str) -> List[str]:
    """用 DB 詞典 + jieba 斷詞抽取可機器讀的關鍵字（去重保序）。"""
    ing_set = build_ingredient_set_from_db()
    # 正規化常見連接符號：+、｜、/、和 → 空白
    normalized = re.sub(r"[+|｜/和]", " ", text or "")
    tokens = [t.strip() for t in jieba.lcut(normalized) if t.strip()]
    filt = [t for t in tokens if t in ing_set]
    seen, uniq = set(), []
    for t in filt:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _keywords_for_intent(intent_out: str, text: str) -> List[str]:
    ing_kws = _extract_keywords_via_db_jieba(text)

    if intent_out == "nutrition":
        # 不再自動加「營養」，讓 orchestrator/LLM 來決定
        return ing_kws

    kws = list(ing_kws)
    if intent_out == "price":
        if "價格" not in kws:
            kws.append("價格")
    elif intent_out == "當季蔬菜月份":
        if "產季" not in kws:
            kws.append("產季")
    return kws


def _map_web_results_to_title_link(raw: Any, limit: int = 8) -> List[Dict[str, str]]:
    """
    將 WebSearchTool 的結果映射成 [{title, link}] 陣列。
    盡量容錯（不同工具實作可能 key 名不同）。
    """
    results = []
    items = []
    if isinstance(raw, dict):
        items = raw.get("results") or raw.get("hits") or raw.get("data") or []
    elif isinstance(raw, list):
        items = raw
    for r in items:
        if not isinstance(r, dict):
            continue
        title = (
            r.get("title")
            or r.get("name")
            or r.get("header")
            or r.get("headline")
            or ""
        )
        link = r.get("link") or r.get("url") or r.get("href") or ""
        title = str(title).strip()
        link = str(link).strip()
        if title and link:
            results.append({"title": title, "link": link})
        if len(results) >= limit:
            break
    return results


def _merge_keywords(ingredients: List[str], llm_kws: List[str]) -> List[str]:
    """
    合併本地食材 + LLM 關鍵字：
    - 食材優先（放前面）
    - 去重去空
    - 移除籠統詞「營養」
    """
    out, seen = [], set()
    for t in (ingredients or []) + (llm_kws or []):
        s = str(t).strip()
        if not s:
            continue
        if s == "營養":
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_bt(payload: Dict[str, Any]) -> Dict[str, str] | None:
    r = payload.get("bt_router_span_id")
    s = payload.get("bt_subrouter_span_id")
    if r or s:
        return {"router_span_id": r, "subrouter_span_id": s}
    return None


def _normalize_nutri_term(k: str) -> str:
    """把常見的營養詞變體正規化（僅在 intent=nutrition 使用）。"""
    s = str(k).strip().lower()
    # 水分
    if s in {"水份", "含水量", "含水率", "moisture"}:
        return "水分"
    # 熱量
    if s in {"卡路里", "calorie", "calories", "kcal"}:
        return "熱量"
    # 蛋白質
    if s in {"蛋白", "protein", "proteins"}:
        return "蛋白質"
    return str(k).strip()


def _postprocess_keywords(
    intent_out: str, keywords: List[str], ingredient_kws: List[str]
) -> List[str]:
    """
    非食譜關鍵字收斂（nutrition 強化版）：
    - 移除「營養」
    - 正規化：水份/含水量/含水率/moisture → 水分；卡路里/calorie/calories/kcal → 熱量；蛋白/protein → 蛋白質
    - 若同時有「蛋白質」與「蛋白」→ 留蛋白質
    - 若已有具體維他命X → 移除泛稱「維他命」「維生素」
    - 對營養類詞做子字串去重（不動食材）；若同時有「水」與「水分」→ 刪「水」
    - 去重保序
    """
    # 先移除「營養」
    ks = [k for k in keywords if k and k != "營養"]

    if intent_out == "nutrition":
        # 正規化常見變體
        ks = [_normalize_nutri_term(k) for k in ks]

        # 蛋白 ↔ 蛋白質：以蛋白質為準
        if "蛋白質" in ks and "蛋白" in ks:
            ks = [k for k in ks if k != "蛋白"]

        # 有具體維他命X → 移除泛稱
        has_specific_vit = any(
            k.startswith("維他命") and len(k) > 3 for k in ks
        ) or any(k.startswith("維生素") and len(k) > 3 for k in ks)
        if has_specific_vit:
            ks = [k for k in ks if k not in ("維他命", "維生素")]

        # 子字串去重（僅對營養詞）；同時處理「水」vs「水分」
        def _is_nutri(k: str) -> bool:
            return (
                k
                in {
                    "蛋白",
                    "蛋白質",
                    "熱量",
                    "水分",
                    "鈣",
                    "鐵",
                    "鋅",
                    "鎂",
                    "鈉",
                    "鉀",
                    "磷",
                    "硒",
                    "葉酸",
                }
                or k.startswith("維他命")
                or k.startswith("維生素")
                or k.endswith("質")
            )

        final = []
        for k in ks:
            # 刪掉「水」這種太泛的單字（如果同時有水分）
            if k == "水" and "水分" in ks:
                continue
            # 營養詞：若是別的更長 token 的子字串 → 移除
            if _is_nutri(k) and any((k != o and k in o) for o in ks):
                continue
            final.append(k)
        ks = final

    # 去重保序
    seen, out = set(), []
    for k in ks:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# ---- API: 單一路由，永遠回傳舊格式（intent + payload） ----------------------------


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    """
    規格（固定舊格式）：
    - price/nutrition/當季蔬菜月份 → {"intent": <...>, "payload": {"keywords": [..]}}
    - recipe → {"intent": "recipe", "payload": {"summary_text": "..."}}
      * 若 orchestrator 告知 web_results（DB 查無 → 自動網搜），則回：
        {"intent": "web_search", "payload": {"results": [{"title": "...","link": "..."}]}}
    - web_search_only → {"intent": "web_search", "payload": {"results": [{"title": "...","link": "..."}]}}
    """
    # 1) 純網路搜尋模式（不進 orchestrator）
    if (req.mode or "").lower() == "web_search_only":
        q = (req.prompt or "").strip()
        if not q:
            return RouteResponse(intent="web_search", payload={"results": []})
        if not WebSearchTool:
            return RouteResponse(intent="web_search", payload={"results": []})
        tool = WebSearchTool()
        raw = (
            tool.run(
                WebSearchInput(query_text=q, top_k=min(req.top_k, 10)).model_dump()
            )
            if WebSearchInput
            else tool.run({"query_text": q, "top_k": min(req.top_k, 10)})
        )
        results = _map_web_results_to_title_link(raw, limit=min(req.top_k, 10))
        return RouteResponse(intent="web_search", payload={"results": results})

    # 2) 一般路徑：走 orchestrator
    text = (req.text or "").strip()
    if not text:
        return RouteResponse(intent="other", payload={"keywords": []})

    payload = run_orchestrator(text, req.top_k)
    intent_out = _normalize_intent_to_output(payload.get("intent", "other"))

    if intent_out == "recipe":
        web_results = payload.get("web_results") or []
        bt = _extract_bt(payload)
        if isinstance(web_results, list) and web_results:
            out = {"results": web_results}
            if bt:
                out["bt"] = bt
            return RouteResponse(intent="web_search", payload=out)

        summary = payload.get("summary_text") or ""
        out = {"summary_text": summary}
        if bt:
            out["bt"] = bt
        return RouteResponse(intent="recipe", payload=out)

    # ⭐ 非食譜：合併「食材詞」+「LLM keywords」
    ing_kws = _extract_keywords_via_db_jieba(text)  # 會可靠抓出「高麗菜」這類食材
    kw_from_orch = (
        payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
    )
    keywords = _merge_keywords(ing_kws, kw_from_orch)

    # ⭐ 新增：做營養類去重與子字串清理（不影響食材）
    keywords = _postprocess_keywords(intent_out, keywords, ingredient_kws=ing_kws)

    return RouteResponse(intent=intent_out, payload={"keywords": keywords})
