# scripts/langchain_project/orchestrator_graph.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain.schema import HumanMessage, SystemMessage
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
from scripts.langchain_project.tools.constraints_filter_tool import (  # noqa
    ConstraintsFilterTool,
    ConstraintsInput,
)
from scripts.langchain_project.tools.get_recipe_tool import (  # noqa
    GetRecipeInput,
    GetRecipeTool,
)
from scripts.langchain_project.tools.recipe_retriever_tool import (  # ← 新增：全文檢索  # noqa
    RecipeRetrieverInput,
    RecipeRetrieverTool,
)
from scripts.langchain_project.tools.recipe_search_tool import (  # 食材/標籤 → 重排  # noqa
    RecipeSearchInput,
    RecipeSearchTool,
)

# ---------- 常量 / Prompt 讀取 ----------
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ROUTER_PROMPT = (PROMPTS_DIR / "router_prompt.txt").read_text(encoding="utf-8")
SUB_ROUTER_PROMPT = (PROMPTS_DIR / "sub_router_prompt.txt").read_text(encoding="utf-8")
PRESENTER_PROMPT = (PROMPTS_DIR / "presenter_prompt.txt").read_text(encoding="utf-8")


# ---------- Graph 的共享狀態 ----------
class GraphState(TypedDict, total=False):
    text: str
    top_k: int
    # 由 Router / Sub-router 產生的解析物
    intent: str
    subroute: Dict[str, Any]
    # 中間結果
    candidate_ids: List[int]
    results: List[Dict[str, Any]]  # 由 tools 回傳的精簡結果
    # 最終呈現
    present: Dict[str, Any]
    error: str


# ---------- 節點實作 ----------
def node_router(state: GraphState) -> GraphState:
    """第一層 Router：LLM 產生 IntentOut（結構化）"""
    llm = get_chat_model("router").with_structured_output(IntentOut)
    messages = [SystemMessage(content=ROUTER_PROMPT.replace("{text}", state["text"]))]
    out = llm.invoke(messages)
    return {"intent": out.intent}


def node_sub_router(state: GraphState) -> GraphState:
    """第二層 Sub-router：抽出 sub_intent / ingredients / constraints / name_query"""
    llm = get_chat_model("sub_router").with_structured_output(RecipeRouteOut)
    messages = [
        SystemMessage(content=SUB_ROUTER_PROMPT.replace("{text}", state["text"]))
    ]
    out = llm.invoke(messages)

    # 你的舊 prompt 可能用中文 → 做一次映射保險
    mapping = {
        "食譜查詢": "by_ingredients",
        "特殊需求": "by_constraints",
        "食譜名稱": "by_name",
    }
    sub = out.dict()
    sub["sub_intent"] = mapping.get(sub.get("sub_intent"), sub.get("sub_intent"))
    return {"subroute": sub}


def node_constraints(state: GraphState) -> GraphState:
    """特殊需求過濾：呼叫 constraints_filter_tool"""
    sub = state.get("subroute", {})
    cons = sub.get("constraints", {}) or {}
    tool = ConstraintsFilterTool()
    payload = ConstraintsInput(
        base_ids=[],
        diet=cons.get("diet") if cons.get("diet") != "none" else None,
        no_pork=bool(cons.get("no_pork", False)),
        extra_exclude=cons.get("extra_exclude", []),
        top_k=max(50, state.get("top_k", 5) * 20),
    )
    resp = tool.run(payload.model_dump())
    return {"candidate_ids": resp.get("ids", [])}


def node_search(state: GraphState) -> GraphState:
    """
    依 sub_intent 走對應工具：
    - by_name        → RecipeRetrieverTool（pgvector 全文檢索）
    - by_ingredients → RecipeSearchTool（食材/標籤 OR → 重排）
    - by_constraints → RecipeSearchTool + candidate_ids 交集
    """
    sub = state.get("subroute", {}) or {}
    sub_intent = sub.get("sub_intent", "by_ingredients")
    top_k = state.get("top_k", 5)

    if sub_intent == "by_name":
        q = sub.get("name_query") or state["text"]
        tool = RecipeRetrieverTool()
        payload = RecipeRetrieverInput(
            query_text=q,
            top_k=top_k,
            candidate_ids=[],  # 若想限制在 constraints 結果內，可改 state.get("candidate_ids", [])
        )
        resp = tool.run(payload.model_dump())
        return {"results": resp.get("results", [])}

    # 其餘兩種 → 走食材/標籤搜尋
    user_text = state["text"]
    if sub_intent == "by_name" and sub.get("name_query"):
        user_text = sub["name_query"]

    tool = RecipeSearchTool()
    payload = RecipeSearchInput(
        user_text=user_text,
        top_k=top_k,
        candidate_ids=(
            state.get("candidate_ids", []) if sub_intent == "by_constraints" else []
        ),
    )
    resp = tool.run(payload.model_dump())
    return {"results": resp.get("results", [])}


def node_presenter(state: GraphState) -> GraphState:
    """Presenter：將精簡欄位交給 LLM 產生條列（最多 top_k 條）"""
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
        ],
        summary_text=txt,
    )
    return {"present": present.dict()}


def node_non_recipe_presenter(state: GraphState) -> GraphState:
    """非 recipe：不查資料，直接回 intent 與簡單關鍵詞"""
    s = state["text"]
    for ch in "，。!？?、；;：:（）()[]【】「」《》<>/\\|":
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
    )
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

    def route_decider(state: GraphState):
        return "sub_router" if state.get("intent") == "recipe" else "present_non_recipe"

    g.add_conditional_edges(
        "router",
        route_decider,
        {"sub_router": "sub_router", "present_non_recipe": "present_non_recipe"},
    )

    def sub_decider(state: GraphState):
        sub = (state.get("subroute") or {}).get("sub_intent", "by_ingredients")
        return "constraints" if sub == "by_constraints" else "search"

    g.add_conditional_edges(
        "sub_router", sub_decider, {"constraints": "constraints", "search": "search"}
    )

    g.add_edge("constraints", "search")
    g.add_edge("search", "present")
    g.add_edge("present", END)
    g.add_edge("present_non_recipe", END)
    return g.compile()


_graph = build_graph()


def run_orchestrator(text: str, top_k: int = 5) -> Dict[str, Any]:
    """對外 API：輸入原始用戶句子，回傳 Presenter 結果 dict。"""
    state: GraphState = {"text": text, "top_k": max(1, min(int(top_k or 5), 10))}
    out = _graph.invoke(state)
    return out.get(
        "present", {"intent": "other", "items": [], "summary_text": "（無輸出）"}
    )
