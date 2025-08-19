from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RecipeHit(BaseModel):
    id: int
    tag: Optional[str] = None
    vege_id: Optional[str] = None
    score: float
    recipe: Dict[str, Any]  # 直接包完整食譜結構


class RouteRequest(BaseModel):
    text: str
    top_k: int = 5


class RouteResponse(BaseModel):
    intent: str
    payload: Dict[str, Any]
