# scripts/langchain_project/APIservice/agent_service.py
"""
第一層 Router（intent）保持不變。
當 intent == "recipe" 時，新增第二層 Sub-Router：
- 食譜查詢：走 Jieba → OR → bge-m3 → DB 補全 → Presenter
- 特殊需求：目前做「偵測與轉交」；回覆簡訊息（之後接你同伴的服務）
- 食譜名稱：以 SQL 模糊/相似度查 top_k，再補全 → Presenter
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.database.ingredient_utils import build_ingredient_set_from_db
from scripts.langchain_project.APIservice.recipe_name_lookup import (
    search_by_recipe_name,
)
from scripts.langchain_project.model import get_chat_model
from scripts.RAG import search_engine

load_dotenv()

# 讀取提示詞
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
ROUTER_PROMPT = (PROMPTS_DIR / "router_prompt.txt").read_text(encoding="utf-8")
PRESENTER_PROMPT = (PROMPTS_DIR / "presenter_prompt.txt").read_text(encoding="utf-8")
SUB_ROUTER_PROMPT = (PROMPTS_DIR / "sub_router_prompt.txt").read_text(encoding="utf-8")

# LLM
llm_router = get_chat_model("router")
llm_sub_router = get_chat_model("sub_router")
llm_presenter = get_chat_model("presenter")


# 型別
class Route(TypedDict):
    intent: Literal["recipe", "nutrition", "price", "identify", "other", "當季蔬菜月份"]


class SubRoute(TypedDict):
    sub_intent: Literal["食譜查詢", "特殊需求", "食譜名稱"]
    name_query: Optional[str]
    ingredients: List[str]
    constraints: Dict[str, object]  # diet/no_pork/no_beef/no_seafood
    reason: str


def _json_safe(s: str) -> Dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def _normalize_topk(k: int) -> int:
    try:
        k = int(k)
    except Exception:
        k = 5
    return max(1, min(10, k))


# ==== 第一層：意圖判斷 ====
def _classify_intent(text: str) -> str:
    # 只替換 {text}，不要用 .format 以免吃掉 JSON 的大括號
    prompt = ROUTER_PROMPT.replace("{text}", text)
    msg = [SystemMessage(content=prompt)]
    resp = llm_router.invoke(msg)
    data = _json_safe(resp.content)
    intent = (data.get("intent") or "other").strip()
    if intent in ("seasonal", "season", "當季蔬菜月份"):
        return "當季蔬菜月份"
    return intent


# ==== 第二層：子路由 ====
def _classify_sub_intent(text: str) -> SubRoute:
    prompt = SUB_ROUTER_PROMPT.replace("{text}", text)
    msg = [SystemMessage(content=prompt)]
    resp = llm_sub_router.invoke(msg)
    data = _json_safe(resp.content)
    sub = {
        "sub_intent": data.get("sub_intent", "食譜查詢"),
        "name_query": data.get("name_query"),
        "ingredients": data.get("ingredients") or [],
        "constraints": data.get("constraints")
        or {"diet": "none", "no_pork": False, "no_beef": False, "no_seafood": False},
        "reason": data.get("reason", ""),
    }
    return sub  # type: ignore


# ==== Presenter ====
def _present_summary_text(user_text: str, recipes: List[Dict]) -> str:
    # 將命中結果濃縮成一段易讀文字（條列、<=5筆）
    # 為避免 prompt 過長，只抽部分欄位
    items = []
    for r in recipes[:5]:
        rec = r.get("recipe", {})
        title = (rec.get("recipe") or rec.get("食譜名稱") or "").strip()
        ingredients = rec.get("ingredients") or []
        ingredients_str = "、".join(
            (i.get("ingredient") or i.get("ingredient_name") or "").strip()
            for i in ingredients[:8]
            if (i.get("ingredient") or i.get("ingredient_name"))
        )
        items.append(
            {
                "id": r.get("id"),
                "title": title,
                "ingredients": ingredients_str,
                "score": float(r.get("score", 0.0)),
            }
        )
    payload = json.dumps({"query": user_text, "items": items}, ensure_ascii=False)

    msg = [
        SystemMessage(content=PRESENTER_PROMPT),
        HumanMessage(content=payload),
    ]
    resp = llm_presenter.invoke(msg)
    return resp.content.strip()


# ==== 對外主函式 ====
def route_and_struct(text: str, top_k: int = 5) -> Dict:
    """
    回傳格式：
    - 若 intent != recipe:
        {"intent": intent, "payload": {"keywords": [...]}}
    - 若 intent == recipe:
        {"intent": "recipe", "payload": {"summary_text": "..."}}
    """
    k = _normalize_topk(top_k)

    # 第一層 Router
    intent = _classify_intent(text)

    if intent != "recipe":
        # 非食譜：抽關鍵字（簡易）
        kw = _extract_keywords_zh(text)
        return {"intent": intent, "payload": {"keywords": kw}}

    # 第二層 Sub-Router
    sub = _classify_sub_intent(text)
    sub_intent = sub["sub_intent"]

    if sub_intent == "食譜名稱":
        q = (sub.get("name_query") or "").strip() or text.strip()
        name_rows = search_by_recipe_name(q, limit=k)
        hits: List[Dict] = []
        for row in name_rows:
            full = search_engine.get_recipe_by_id(row["id"])
            if not full:
                continue
            hits.append(
                {
                    "id": int(row["id"]),
                    "tag": None,
                    "vege_name": full.get("vege_name"),
                    "score": 0.0,  # 名稱查詢暫不使用分數
                    "recipe": full,
                }
            )
        if not hits:
            # 找不到就降級到「食譜查詢」
            sub_intent = "食譜查詢"
        else:
            summary_text = _present_summary_text(text, hits)
            return {"intent": "recipe", "payload": {"summary_text": summary_text}}

    if sub_intent == "特殊需求":
        # 暫時先回一段說明文字，後續你們接你同伴的服務再換掉
        cons = sub.get("constraints") or {}
        pretty = []
        if cons.get("diet") and cons.get("diet") != "none":
            pretty.append(f"飲食：{cons['diet']}")
        for kf, zh in [
            ("no_pork", "不含豬肉"),
            ("no_beef", "不含牛肉"),
            ("no_seafood", "不含海鮮"),
        ]:
            if cons.get(kf):
                pretty.append(zh)
        tip = "；".join(pretty) if pretty else "（未指定條件）"
        summary_text = f"已識別為「特殊需求」：{tip}\n此路徑將與你的同伴服務銜接，暫時僅回偵測結果。"
        return {"intent": "recipe", "payload": {"summary_text": summary_text}}

    # 預設：食譜查詢
    ing_set = build_ingredient_set_from_db()
    tokens = search_engine.pull_ingredients(text, ing_set)
    hits = search_engine.tag_then_vector_rank(text, tokens_from_jieba=tokens, top_k=k)
    summary_text = _present_summary_text(text, hits) if hits else "找不到符合的食譜。"
    return {"intent": "recipe", "payload": {"summary_text": summary_text}}


# ==== 小工具 ====
_ZH_SPLIT = "，。！!？?、；;：:（）()[]【】「」《》<>/\\|"


def _extract_keywords_zh(text: str, max_kw: int = 3) -> List[str]:
    """非常輕量的中文詞切；只是給非 recipe 類型當作 keywords。"""
    s = (text or "").strip()
    for ch in _ZH_SPLIT:
        s = s.replace(ch, " ")
    toks = [t for t in s.split() if 1 < len(t) <= 8]
    # 去重並截斷
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= max_kw:
            break
    return out
