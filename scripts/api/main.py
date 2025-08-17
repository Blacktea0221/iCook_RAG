# scripts/api/main.py
import os
import sys
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI

# 讓 scripts 成為可 import 路徑（沿用你原本的做法）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from api.schemas import RouteRequest, RouteResponse  # payload 是 Dict，可容納我們的新結構
from scripts.langchain_project.APIservice.agent_service import route_and_struct

load_dotenv()

app = FastAPI(title="Agent API (Router + Recipe RAG + Presenter)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    """
    - 食譜：回 {"intent":"recipe","payload":{"summary_text":"文字摘要","hits":[...]}}
    - 其他：回 {"intent":"other|nutrition|price|identify","payload":{"keywords":[...]} }
    """
    result: Dict = route_and_struct(req.text, top_k=req.top_k)
    return RouteResponse(intent=result["intent"], payload=result["payload"])
