# scripts/langchain_project/orchestrator_graph.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
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


def _has_month_or_season_trigger(text: str) -> bool:
    t = (text or "").lower()
    # 1~12 月（阿拉伯數字）
    if re.search(r"(^|\s)([1-9]|1[0-2])\s*月", t):
        return True
    # 中文數字月份 & 常見語彙
    month_words = [
        "一月",
        "二月",
        "三月",
        "四月",
        "五月",
        "六月",
        "七月",
        "八月",
        "九月",
        "十月",
        "十一月",
        "十二月",
        "本月",
        "這個月",
        "下月",
        "上月",
        "春天",
        "夏天",
        "秋天",
        "冬天",
        "春季",
        "夏季",
        "秋季",
        "冬季",
        "q1",
        "q2",
        "q3",
        "q4",
        "第一季",
        "第二季",
        "第三季",
        "第四季",
        "產季",
        "當季",
        "盛產",
        "季節",
        "時令",
        "上市",
        "上市時間",
        "幾月",
        "月份",
        "何時",
    ]
    return any(w in t for w in month_words)


def _has_nutrition_trigger(text: str) -> bool:
    t = (text or "").lower()
    keys = [
        "熱量",
        "卡路里",
        "kcal",
        "蛋白質",
        "protein",
        "脂肪",
        "碳水",
        "膳食纖維",
        "維他命",
        "維生素",
        "鈣",
        "鐵",
        "鋅",
        "鎂",
        "鉀",
        "鈉",
        "磷",
        "硒",
        "葉酸",
        "水分",
        "含水量",
        "含水率",
        "moisture",
    ]
    return any(w in t for w in keys)


def _has_price_trigger(text: str) -> bool:
    t = (text or "").lower()
    keys = [
        "價格",
        "價錢",
        "多少錢",
        "幾元",
        "幾塊",
        "nt",
        "元",
        "比價",
        "行情",
        "行情報價",
        "一斤多少",
        "一公斤多少",
        "菜價",
        "漲價",
        "上漲",
        "下跌",
        "降價",
        "調漲",
        "調降",
        "便宜",
        "貴",
    ]
    return any(k in t for k in keys)


def _has_constraint_trigger(text: str) -> bool:
    t = (text or "").lower()
    return any(
        k in t
        for k in [
            "素食",
            "蔬食",
            "純素",
            "全素",
            "蛋奶素",
            "vegan",
            "vegetarian",
            "不吃豬",
            "不要豬",
            "無豬",
            "不含豬",
            "禁豬",
            "no pork",
            "清真",
            "halal",
            "不要培根",
            "不吃培根",
        ]
    )


def _looks_like_dish_name(text: str) -> bool:
    t = (text or "").lower().strip()
    # 常見料理/食品後綴（中/英文）
    pat = r"(炒|湯|煎|炸|烤|拌|燉|羹|飯|麵|粥|餅|餃|捲|沙拉|醬|果醬|滷|燜|焗|涼拌|醃)$"
    if re.search(pat, t):
        return True
    eng = [
        "sauce",
        "jam",
        "soup",
        "salad",
        "noodles",
        "rice",
        "stew",
        "stir-fry",
        "stir fry",
    ]
    return any(w in t for w in eng)


def node_router(state: GraphState) -> GraphState:
    text = state["text"]
    llm = get_chat_model("router")
    messages = [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=text)]
    try:
        out = llm.invoke(messages).content
    except Exception:
        # LLM 失敗時：優先級 nutrition > price > seasonal > recipe > other
        if _has_nutrition_trigger(text):
            return {"intent": "nutrition"}
        if _has_price_trigger(text):
            return {"intent": "price"}
        if _has_month_or_season_trigger(text):
            return {"intent": "seasonal"}
        if _has_constraint_trigger(text) or _looks_like_dish_name(text):
            return {"intent": "recipe"}
        return {"intent": "other"}

    intent = "other"
    try:
        intent = IntentOut.model_validate_json(out).intent
    except Exception:
        intent = "other"

    # ---- 覆寫保底（固定優先序）----
    # 1) 有營養/價格觸發：一定分到 nutrition / price
    if _has_nutrition_trigger(text):
        intent = "nutrition"
    elif _has_price_trigger(text):
        intent = "price"
    # 2) 有月份/季節觸發：一定分到 seasonal
    elif _has_month_or_season_trigger(text):
        intent = "seasonal"
    # 3) 有飲食限制（素食/不吃豬…）或看起來像一道料理/醬料名：分到 recipe
    elif _has_constraint_trigger(text) or _looks_like_dish_name(text):
        intent = "recipe"

    return {"intent": intent}


def _detect_constraints_from_text(text: str) -> dict:
    t = (text or "").lower()
    diet = None
    # 素食相關（可再擴充）
    if any(
        k in t
        for k in ["素食", "蔬食", "純素", "全素", "vegan", "vegetarian", "蛋奶素"]
    ):
        diet = "vegetarian"
    # 禁豬/清真
    no_pork = any(
        k in t
        for k in [
            "不吃豬",
            "不要豬",
            "無豬",
            "不含豬",
            "禁豬",
            "no pork",
            "清真",
            "halal",
            "不要培根",
            "不吃培根",
        ]
    )
    return {"diet": diet, "no_pork": no_pork}


def node_sub_router(state: GraphState) -> GraphState:
    if state.get("intent") != "recipe":
        return {}

    import json

    text = state["text"]
    llm = get_chat_model("sub_router")
    messages = [SystemMessage(content=SUB_ROUTER_PROMPT), HumanMessage(content=text)]
    raw = llm.invoke(messages).content
    try:
        sub = json.loads(raw)
    except Exception:
        sub = {"sub_intent": "食材查詢", "ingredients": [], "constraints": {}}

    # ⛳ 強制覆寫：只要句子含有素食/禁豬等訊號 → 一律切到「特殊需求」，並補上 constraints
    auto = _detect_constraints_from_text(text)
    if auto.get("diet") or auto.get("no_pork"):
        sub["sub_intent"] = "特殊需求"
        cons = sub.get("constraints") or {}
        if auto.get("diet") and not cons.get("diet"):
            cons["diet"] = auto["diet"]
        if auto.get("no_pork") and not cons.get("no_pork"):
            cons["no_pork"] = True
        sub["constraints"] = cons

    return {"subroute": sub}


def node_constraints(state: GraphState) -> GraphState:
    sub = state.get("subroute") or {}
    if not sub or sub.get("sub_intent") != "特殊需求":
        return {}

    cons = sub.get("constraints") or {}
    tool = ConstraintsFilterTool()

    # 第一次：依使用者限制（可能有 diet=vegetarian 與/或 no_pork）
    try:
        out = tool.run(
            {
                "base_ids": [],
                "diet": cons.get("diet"),
                "no_pork": bool(cons.get("no_pork")),
                "extra_exclude": [],  # ← 修正欄位名（原本誤寫 keywords_exclude）
                "top_k": 500,
            }
        )
    except Exception:
        out = {"ids": [], "reason": "", "error": "constraints tool error"}

    ids = list(out.get("ids") or [])

    # 若要求素食卻無結果、且有 no_pork，退一步：只禁豬
    if not ids and cons.get("no_pork"):
        try:
            out2 = tool.run(
                {
                    "base_ids": [],
                    "diet": None,  # 放寬素食標記
                    "no_pork": True,
                    "extra_exclude": [],
                    "top_k": 500,
                }
            )
            ids = list(out2.get("ids") or [])
        except Exception:
            pass

    return {"candidate_ids": ids, "constraints": cons}


def node_search(state: GraphState) -> GraphState:
    sub = state.get("subroute") or {}
    cons = state.get("constraints") or {}
    k = max(1, min(int(state.get("top_k", 5)), 20))

    def _unwrap(res):
        if isinstance(res, dict):
            return res.get("results") or []
        return res or []

    def _passes_constraints(item: Dict[str, Any]) -> bool:
        title = (item.get("title") or "").lower()
        tags = [str(t).lower() for t in (item.get("preview_tags") or [])]

        banned_meats = {
            "培根",
            "豬",
            "豬肉",
            "五花肉",
            "火腿",
            "肉燥",
            "香腸",
            "絞肉",
            "牛",
            "牛肉",
            "羊",
            "雞",
            "雞肉",
            "鴨",
            "鵝",
            "魚",
            "鮭",
            "鯖",
            "鮪",
            "蝦",
            "蟹",
            "花枝",
            "章魚",
            "蛤蜊",
            "牡蠣",
        }
        if cons.get("no_pork"):
            if any(w in title for w in ["培根", "豬", "豬肉", "五花肉", "火腿"]) or any(
                any(w in t for w in ["培根", "豬", "豬肉", "五花肉", "火腿"])
                for t in tags
            ):
                return False
        if cons.get("diet") == "vegetarian":
            if any(w in title for w in banned_meats) or any(
                any(w in t for w in banned_meats) for t in tags
            ):
                return False
        return True

    # 1) 食譜名稱：全文檢索（同樣可吃候選集）
    if sub.get("sub_intent") == "食譜名稱":
        name_q = (sub.get("name_query") or "").strip()
        if not name_q:
            return {"results": []}
        tool = RecipeRetrieverTool()
        res = tool.run(
            {
                "query_text": name_q,
                "top_k": k,
                "candidate_ids": state.get("candidate_ids") or [],
            }
        )
        items = _unwrap(res)
        if cons:
            items = [it for it in items if _passes_constraints(it)]
        return {"results": items}

    # 2) 特殊需求：優先在候選集合內召回；若空，降級一般召回+內容過濾
    if sub.get("sub_intent") == "特殊需求":
        tool = RecipeSearchTool()
        first = tool.run(
            {
                "user_text": state["text"],
                "top_k": k,
                "candidate_ids": state.get("candidate_ids") or [],
            }
        )
        items = _unwrap(first)

        if not items:
            general = tool.run(
                {
                    "user_text": state["text"],
                    "top_k": max(k * 2, 10),  # 多抓一些，過濾後再截 k
                    "candidate_ids": [],
                }
            )
            items = _unwrap(general)

        if cons:
            items = [it for it in items if _passes_constraints(it)]
        return {"results": items[:k]}

    # 3) 食材查詢（預設）：也吃 candidate_ids（若先前有產生）
    tool = RecipeSearchTool()
    res = tool.run(
        {
            "user_text": state["text"],
            "top_k": k,
            "candidate_ids": state.get("candidate_ids") or [],
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
