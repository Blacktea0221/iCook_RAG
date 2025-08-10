import os
import sys
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI

# 讓 scripts 成為可 import 路徑
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from api.router.intent_router import classify_intent
from api.schemas import RecipeHit, RouteRequest, RouteResponse
from RAG import search_engine

load_dotenv()

app = FastAPI(title="Agent API (Router + Recipe RAG)")


@app.get("/health")
def health():
    return {"status": "ok"}


def _recipe_rag(query: str, top_k: int) -> List[Dict]:
    # 從 DB 準備 ingredient_set（用你的 main.py 方法）
    from main import build_ingredient_set_from_db

    ing_set = build_ingredient_set_from_db()
    tokens = search_engine.pull_ingredients(query, ing_set)
    results = search_engine.tag_then_vector_rank(
        query, tokens_from_jieba=tokens, top_k=top_k
    )
    return results


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    intent = classify_intent(req.text)
    if intent == "recipe":
        hits = _recipe_rag(req.text, req.top_k)
        # 直接回傳完整 RAG 結果
        return RouteResponse(intent=intent, payload={"hits": hits})
    return RouteResponse(intent="other", payload={"message": "目前僅支援食譜查詢。"})
