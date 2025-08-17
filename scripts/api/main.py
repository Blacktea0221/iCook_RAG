# scripts/api/main.py
import os
import sys
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from api.schemas import RouteRequest, RouteResponse  # 【已存在】請求/回應模型

from scripts.langchain_project.APIservice.agent_service import (
    route_and_struct,
)  # 【你現有的分類+執行總管】

load_dotenv()
app = FastAPI(title="Agent API (Router + Recipe RAG + Presenter)")


@app.get("/health")
def health():
    return {"status": "ok"}


def _normalize_intent(x: str) -> str:
    x = (x or "").strip().lower()
    # 支援 LLM 回 "seasonal"/"當季蔬菜月份" 的同義
    if x in ("seasonal", "season", "當季蔬菜月份"):
        return "當季蔬菜月份"
    elif x in ("recipe", "nutrition", "price", "other"):
        return x
    return "other"


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    """
    需求：
    - 仍由 LLM 先做分類。
    - 只有 recipe 才執行 RAG + Presenter，並只回 summary_text。
    - 其他分類不執行你的食譜程式，直接把分類回給 LINE Bot 使用。
    """
    # 呼叫你原本的服務（裡面會做 LLM 分類）
    result: Dict = route_and_struct(
        req.text, top_k=req.top_k
    )  # 這行是原樣保留:contentReference[oaicite:2]{index=2}
    intent_raw = result.get("intent", "other")
    intent = _normalize_intent(intent_raw)

    if intent == "recipe":
        # 只回 Presenter 做好的條列摘要；不帶 hits
        payload = {"summary_text": result.get("payload", {}).get("summary_text", "")}
        return RouteResponse(intent="recipe", payload=payload)

    # 非 recipe → 不執行你的 RAG 程式，僅回分類給 LINE Bot 使用
    # 為方便 LINE 端顯示/除錯，可以保留輕量 keywords（若有）
    keywords = result.get("payload", {}).get("keywords", [])
    return RouteResponse(intent=intent, payload={"keywords": keywords})
