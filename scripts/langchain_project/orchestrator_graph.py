# scripts/langchain_project/orchestrator_graph.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from scripts.langchain_project.model import get_chat_model
from scripts.langchain_project.orchestrator_schemas import (
    IntentOut,
    PresentItem,
    PresentOut,
)

# ⛳ 已移除：recipe_name_lookup 的匯入（不需要）
from scripts.langchain_project.tools.constraints_filter_tool import (
    ConstraintsFilterTool,
)
from scripts.langchain_project.tools.recipe_retriever_tool import RecipeRetrieverTool
from scripts.langchain_project.tools.recipe_search_tool import RecipeSearchTool
from scripts.langchain_project.tools.web_search_tool import (
    WebSearchInput,
    WebSearchTool,
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ROUTER_PROMPT = (PROMPTS_DIR / "router_prompt.txt").read_text(encoding="utf-8")
SUB_ROUTER_PROMPT = (PROMPTS_DIR / "sub_router_prompt.txt").read_text(encoding="utf-8")
PRESENTER_PROMPT = (PROMPTS_DIR / "presenter_prompt.txt").read_text(encoding="utf-8")
NON_RECIPE_KW_PROMPT = (PROMPTS_DIR / "non_recipe_keywords_prompt.txt").read_text(
    encoding="utf-8"
)  # ← 新增


class GraphState(TypedDict, total=False):
    text: str
    top_k: int
    intent: str
    subroute: Dict[str, Any]
    candidate_ids: List[int]
    results: List[Dict[str, Any]]
    present: Dict[str, Any]
    error: str


def node_router(state: GraphState) -> GraphState:
    text = state["text"]
    llm = get_chat_model("router")
    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=text),
    ]
    out = llm.invoke(messages).content
    intent = "other"
    try:
        intent = IntentOut.model_validate_json(out).intent
    except Exception:
        intent = "other"
    return {"intent": intent}


def node_sub_router(state: GraphState) -> GraphState:
    if state.get("intent") != "recipe":
        return {}
    text = state["text"]
    llm = get_chat_model("sub_router")
    messages = [
        SystemMessage(content=SUB_ROUTER_PROMPT),
        HumanMessage(content=text),
    ]
    raw = llm.invoke(messages).content
    try:
        import json

        sub = json.loads(raw)
    except Exception:
        sub = {"sub_intent": "食譜查詢", "ingredients": [], "constraints": {}}
    return {"subroute": sub}


def node_constraints(state: GraphState) -> GraphState:
    sub = state.get("subroute") or {}
    if not sub or sub.get("sub_intent") != "特殊需求":
        return {}
    tool = ConstraintsFilterTool()
    cons = sub.get("constraints") or {}
    try:
        out = tool.run(
            {
                "base_ids": [],
                "diet": cons.get("diet"),
                "no_pork": bool(cons.get("no_pork")),
                "keywords_exclude": [],
            }
        )
    except Exception:
        out = {"ids": [], "reason": "", "error": "constraints tool error"}
    ids = out.get("ids") or []
    return {"candidate_ids": ids}


def node_search(state: GraphState) -> GraphState:
    sub = state.get("subroute") or {}
    k = max(1, min(int(state.get("top_k", 5)), 20))

    def _unwrap(res):
        # 工具可能回 dict 或 list；我們只要列表
        if isinstance(res, dict):
            return res.get("results") or []
        return res or []

    # 1) 食譜名稱：全文檢索（注意參數是 query_text）
    if sub.get("sub_intent") == "食譜名稱":
        name_q = (sub.get("name_query") or "").strip()
        if not name_q:
            return {"results": []}
        tool = RecipeRetrieverTool()
        res = tool.run({"query_text": name_q, "top_k": k, "candidate_ids": []})
        return {"results": _unwrap(res)}

    # 2) 特殊需求：先過濾，再查詢（即使 candidate_ids 空也要帶參數）
    if sub.get("sub_intent") == "特殊需求":
        tool = RecipeSearchTool()
        res = tool.run(
            {
                "user_text": state["text"],
                "top_k": k,
                "candidate_ids": state.get("candidate_ids") or [],
            }
        )
        return {"results": _unwrap(res)}

    # 3) 食材查詢（預設）：一定帶 candidate_ids=[]
    tool = RecipeSearchTool()
    res = tool.run(
        {
            "user_text": state["text"],
            "top_k": k,
            "candidate_ids": [],
        }
    )
    return {"results": _unwrap(res)}


def after_search(state: GraphState) -> str:
    return "present" if (state.get("results") or []) else "web_fallback"


def node_presenter(state: GraphState) -> GraphState:
    raw = state.get("results") or []
    if isinstance(raw, dict):  # ← 防呆：有人傳了整包 dict
        raw = raw.get("results") or []
    k = max(1, state.get("top_k", 5))

    items = [
        {
            "id": r.get("id"),
            "title": r.get("title", ""),
            "preview_tags": r.get("preview_tags", []),
            "score": float(r.get("score", 0.0)),
            "link": r.get("link"),
        }
        for r in (raw[:k] if isinstance(raw, list) else [])
    ]

    payload = {"query": state["text"], "items": items}
    llm = get_chat_model("presenter")
    messages = [
        SystemMessage(content=PRESENTER_PROMPT),
        HumanMessage(content=str(payload)),
    ]
    txt = llm.invoke(messages).content

    present = PresentOut(
        intent=state.get("intent", "recipe"),
        items=[
            PresentItem(
                id=int(i["id"]),
                title=i["title"],
                score=float(i["score"]),
                link=i.get("link"),
            )
            for i in items
            if i.get("id") is not None
        ],
        summary_text=txt,
    )
    return {"present": present.dict()}


def node_web_fallback_search(state: GraphState) -> GraphState:
    """
    查無資料時的網路搜尋 fallback：
    1) 維持 summary_text（原本就有）
    2) 另外產出 web_results = [{title, link}] 供外層（agent_service）辨識並回傳 JSON
    """
    q = state["text"]
    k = max(1, min(int(state.get("top_k", 3)), 10))
    tool = WebSearchTool()
    out = tool.run(WebSearchInput(query_text=q, top_k=k).model_dump())

    web_results: List[Dict[str, str]] = []
    if out.get("status") == "ok" and out.get("results"):
        for r in out["results"]:
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            if title and link:
                web_results.append({"title": title, "link": link})
        lines = [
            f"{idx}. {r['title']}\n{r['link']}" for idx, r in enumerate(web_results, 1)
        ]
        body = "查無資料，為你找了這些網路結果：\n" + "\n".join(lines)
    elif out.get("status") == "empty":
        body = "查無資料，且目前也找不到合適的網路結果。"
    else:
        body = "查無資料，嘗試網路搜尋時發生問題。"

    present = PresentOut(
        intent=state.get("intent", "recipe"),
        items=[],
        summary_text=body,
    ).dict()
    present["web_results"] = web_results
    return {"present": present}


def node_non_recipe_presenter(state: GraphState) -> GraphState:
    """
    非 recipe 類別的關鍵字抽取改由 LLM 依 prompt 產出 JSON：{"keywords":[...]}
    """
    import json

    llm = get_chat_model("router")  # 小模型即可
    payload = {"intent": state.get("intent", "other"), "text": state["text"]}
    messages = [
        SystemMessage(content=NON_RECIPE_KW_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    raw = llm.invoke(messages).content

    keywords: List[str] = []
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("keywords"), list):
            keywords = [str(x).strip() for x in data["keywords"] if str(x).strip()]
    except Exception:
        # 失敗保底：取 2~3 個簡單 token
        s = state["text"]
        for ch in "，。!？?、；;：:（）()[]【】「」《》<>/\\|":
            s = s.replace(ch, " ")
        seen = set()
        for t in s.split():
            if 1 < len(t) <= 12 and t not in seen:
                seen.add(t)
                keywords.append(t)
            if len(keywords) >= 3:
                break

    body = f"已識別為「{state.get('intent','other')}」。"
    present = PresentOut(
        intent=state.get("intent", "other"),
        items=[],
        summary_text=body,
    ).dict()
    present["keywords"] = keywords  # ★ 關鍵：讓外層能拿到 LLM keywords
    return {"present": present}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("router", node_router)
    g.add_node("sub_router", node_sub_router)
    g.add_node("constraints", node_constraints)
    g.add_node("search", node_search)
    g.add_node("present", node_presenter)
    g.add_node("web_fallback", node_web_fallback_search)
    g.add_node("present_non_recipe", node_non_recipe_presenter)

    g.set_entry_point("router")

    def route_decider(state: GraphState):
        return "sub_router" if state.get("intent") == "recipe" else "present_non_recipe"

    g.add_conditional_edges(
        "router",
        route_decider,
        {"sub_router": "sub_router", "present_non_recipe": "present_non_recipe"},
    )
    g.add_edge("sub_router", "constraints")
    g.add_edge("constraints", "search")
    g.add_conditional_edges(
        "search", after_search, {"present": "present", "web_fallback": "web_fallback"}
    )
    g.add_edge("present", END)
    g.add_edge("web_fallback", END)
    g.add_edge("present_non_recipe", END)
    return g.compile()


_graph = build_graph()


def run_orchestrator(text: str, top_k: int = 5) -> Dict[str, Any]:
    state: GraphState = {"text": text, "top_k": max(1, min(int(top_k or 5), 10))}
    out = _graph.invoke(state)
    return out.get(
        "present", {"intent": "other", "items": [], "summary_text": "（無輸出）"}
    )
