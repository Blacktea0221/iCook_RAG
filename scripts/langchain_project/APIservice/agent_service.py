# scripts/langchain_project/APIservice/agent_service.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ⚠️ 先不要動到 Orchestrator，確認健康後再接回
from scripts.api.schemas import RouteRequest, RouteResponse
from scripts.langchain_project.orchestrator_graph import run_orchestrator

app = FastAPI(
    title="iCook RAG Orchestrator",
    # 先用 FastAPI 內建 /docs，避免自訂靜態檔也被誤判
    docs_url="/docs",
    redoc_url="/redoc",
)

# 放寬 CORS，避免外層工具擋住
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 健康檢查與最小頁面（不用任何外部資源） ---
@app.get("/", include_in_schema=False)
def home():
    return {"ok": True, "msg": "iCook RAG API is alive."}


@app.get("/healthz", include_in_schema=False)
def healthz():
    return "ok"


# --- 正式路由（走 Orchestrator） ---
@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest):
    result = run_orchestrator(req.text, req.top_k)
    return RouteResponse(intent=result["intent"], payload=result)
