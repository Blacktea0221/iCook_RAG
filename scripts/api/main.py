# scripts/api/main.py
from typing import Dict

from fastapi import FastAPI

from scripts.api.schemas import RouteRequest, RouteResponse
from scripts.langchain_project.orchestrator_graph import run_orchestrator

app = FastAPI(title="Agent API (Orchestrator)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    result: Dict = run_orchestrator(req.text, req.top_k)
    return RouteResponse(intent=result.get("intent", "other"), payload=result)
