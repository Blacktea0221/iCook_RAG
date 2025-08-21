# scripts/langchain_project/orchestrator_graph.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from scripts.langchain_project.model import (
    get_chat_model,
)  # LLM 工廠（router/sub_router/presenter）  # noqa
from scripts.langchain_project.orchestrator_schemas import (
    IntentOut,
    PresentItem,
    PresentOut,
    RecipeRouteOut,
)
from scripts.langchain_project.tools.constraints_filter_tool import (  # 特殊需求過濾  # noqa
    ConstraintsFilterTool,
    ConstraintsInput,
)
from scripts.langchain_project.tools.get_recipe_tool import (  # 取完整食譜（若需要）  # noqa
    GetRecipeInput,
    GetRecipeTool,
)
from scripts.langchain_project.tools.recipe_search_tool import (  # 查食材/標籤 → 重排  # noqa
    RecipeSearchInput,
    RecipeSearchTool,
)

# ---------- 常量 / Prompt 讀取 ----------
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ROUTER_PROMPT = (PROMPTS_DIR / "router_prompt.txt").read_text(
    encoding="utf-8"
)  # 說明意圖分類規則（輔助判斷）  # noqa
SUB_ROUTER_PROMPT = (PROMPTS_DIR / "sub_router_prompt.txt").read_text(
    encoding="utf-8"
)  # 子意圖抽槽位  # noqa
PRESENTER_PROMPT = (PROMPTS_DIR / "presenter_prompt.txt").read_text(
    encoding="utf-8"
)  # 條列輸出說明  # noqa


# ---------- Graph 的共享狀態 ----------
class GraphState(TypedDict, total=False):
    text: str
    top_k: int
    # 由 Router / Sub-router 產生的解析物
    intent: str
    subroute: Dict[str, Any]
    # 中間結果
    candidate_ids: List[int]
    results: List[
        Dict[str, Any]
    ]  # 由 tools 回傳的精簡結果（RecipeSearchOutput.results）
    # 最終呈現
    present: Dict[str, Any]
    error: str


# ---------- 節點實作 ----------
def node_router(state: GraphState) -> GraphState:
    """
    第一層 Router：LLM 產生 IntentOut（結構化）
    """
    llm = get_chat_model("router").with_structured_output(IntentOut)
    # 用 system 說明規則 + human 放原文，並保留你現有 prompt 的語境
    messages = [SystemMessage(content=ROUTER_PROMPT.replace("{text}", state["text"]))]
    out = llm.invoke(messages)
    return {"intent": out.intent}


def node_sub_router(state: GraphState) -> GraphState:
    """
    第二層 Sub-router（只在 intent=recipe 時跑）：抽出 sub_intent / ingredients / constraints / name_query
    """
    llm = get_chat_model("sub_router").with_structured_output(RecipeRouteOut)
    # Sub-router prompt 仍然用你現有敘述，LLM 以 Pydantic 保證結構
    messages = [
        SystemMessage(content=SUB_ROUTER_PROMPT.replace("{text}", state["text"]))
    ]
    out = llm.invoke(messages)

    # 你的舊 prompt會回中文「食譜查詢/特殊需求/食譜名稱」，所以這裡做一次映射保險
    mapping = {
        "食譜查詢": "by_ingredients",
        "特殊需求": "by_constraints",
        "食譜名稱": "by_name",
    }
    sub = out.dict()
    sub["sub_intent"] = mapping.get(sub.get("sub_intent"), sub.get("sub_intent"))
    return {"subroute": sub}


def node_constraints(state: GraphState) -> GraphState:
    """
    特殊需求過濾：呼叫 constraints_filter_tool
    """
    sub = state.get("subroute", {})
    cons = sub.get("constraints", {}) or {}
    tool = ConstraintsFilterTool()
    payload = ConstraintsInput(
        base_ids=[],  # 若你有上一階段候選，丟這裡；目前直接全庫過濾
        diet=cons.get("diet") if cons.get("diet") != "none" else None,
        no_pork=bool(cons.get("no_pork", False)),
        extra_exclude=cons.get("extra_exclude", []),
        top_k=max(50, state.get("top_k", 5) * 20),
    )
    resp = tool.run(payload.model_dump())
    return {"candidate_ids": resp.get("ids", [])}


def node_search(state: GraphState) -> GraphState:
    """
    食譜檢索：呼叫 recipe_search_tool
    - by_ingredients: 直接用原句 user_text
    - by_constraints: 搭配 candidate_ids 做交集後再重排
    - by_name: 目前先降級為把 name_query 併入 user_text（你之後可換成獨立的 name_lookup_tool）
    """
    sub = state.get("subroute", {}) or {}
    sub_intent = sub.get("sub_intent", "by_ingredients")
    user_text = state["text"]
    if sub_intent == "by_name" and sub.get("name_query"):
        user_text = sub["name_query"]

    tool = RecipeSearchTool()
    payload = RecipeSearchInput(
        user_text=user_text,
        top_k=state.get("top_k", 5),
        candidate_ids=(
            state.get("candidate_ids", []) if sub_intent == "by_constraints" else []
        ),
    )
    resp = tool.run(payload.model_dump())  # dict
    return {"results": resp.get("results", [])}


def node_presenter(state: GraphState) -> GraphState:
    """
    Presenter：組一份 JSON（id/title/score/link），交給 LLM 產生條列文字（最多5條）
    """
    items = [
        {
            "id": r.get("id"),
            "title": r.get("title", ""),
            "preview_tags": r.get("preview_tags", []),
            "score": float(r.get("score", 0.0)),
            "link": r.get("link"),
        }
        for r in (state.get("results") or [])[: max(1, state.get("top_k", 5))]
    ]
    payload = {"query": state["text"], "items": items}
    llm = get_chat_model("presenter")
    messages = [
        SystemMessage(content=PRESENTER_PROMPT),
        HumanMessage(content=str(payload)),
    ]
    txt = llm.invoke(messages).content.strip()

    present = PresentOut(
        intent=state.get("intent", "other"),
        items=[
            PresentItem(
                id=int(i["id"]),
                title=i["title"],
                score=float(i["score"]),
                link=i.get("link"),
            )
            for i in items
        ],  # noqa
        summary_text=txt,
    )
    return {"present": present.dict()}


def node_non_recipe_presenter(state: GraphState) -> GraphState:
    """
    非 recipe 類型：不查資料，直接回 intent 與簡單關鍵詞（可再強化）
    """
    from math import inf

    s = state["text"]
    # 很陽春的中文關鍵詞切分：把標點轉空白再切三個詞
    for ch in "，。！!？?、；;：:（）()[]【】「」《》<>/\\|":
        s = s.replace(ch, " ")
    toks, seen = [], set()
    for t in s.split():
        if 1 < len(t) <= 8 and t not in seen:
            seen.add(t)
            toks.append(t)
        if len(toks) >= 3:
            break
    present = PresentOut(
        intent=state["intent"],
        items=[],
        summary_text=f"已識別為「{state['intent']}」，關鍵詞：{', '.join(toks)}",
    )  # noqa
    return {"present": present.dict()}


# ---------- Graph 組裝 ----------
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("router", node_router)
    g.add_node("sub_router", node_sub_router)
    g.add_node("constraints", node_constraints)
    g.add_node("search", node_search)
    g.add_node("present", node_presenter)
    g.add_node("present_non_recipe", node_non_recipe_presenter)

    g.set_entry_point("router")

    # router → 分流
    def route_decider(state: GraphState):
        return "sub_router" if state.get("intent") == "recipe" else "present_non_recipe"

    g.add_conditional_edges(
        "router",
        route_decider,
        {
            "sub_router": "sub_router",
            "present_non_recipe": "present_non_recipe",
        },
    )

    # sub_router → 依子意圖分流
    def sub_decider(state: GraphState):
        sub = (state.get("subroute") or {}).get("sub_intent", "by_ingredients")
        return "constraints" if sub == "by_constraints" else "search"

    g.add_conditional_edges(
        "sub_router",
        sub_decider,
        {
            "constraints": "constraints",
            "search": "search",
        },
    )

    # constraints → search
    g.add_edge("constraints", "search")
    # search → present
    g.add_edge("search", "present")
    # present / present_non_recipe → END
    g.add_edge("present", END)
    g.add_edge("present_non_recipe", END)
    return g.compile()


_graph = build_graph()


def run_orchestrator(text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    對外 API：輸入原始用戶句子，回傳 PresentOut 的 dict。
    """
    state: GraphState = {"text": text, "top_k": max(1, min(int(top_k or 5), 10))}
    out = _graph.invoke(state)
    return out.get(
        "present", {"intent": "other", "items": [], "summary_text": "（無輸出）"}
    )
