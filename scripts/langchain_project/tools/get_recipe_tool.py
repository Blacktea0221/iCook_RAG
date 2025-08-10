# tools/get_recipe_tool.py
from typing import Dict, Optional

from langchain_core.tools import tool

from scripts.RAG.search_engine import get_recipe_by_id


@tool("get_recipe_by_id", return_direct=False)
def get_recipe_tool(recipe_id: str) -> Optional[Dict]:
    """
    依 ID 讀取完整食譜（title/preview_tags/ingredients/steps）
    """
    rid = str(recipe_id).strip()
    if not rid.isdigit():
        return None
    rec = get_recipe_by_id(int(rid))
    if not rec:
        return None
    return {"id": int(rid), "recipe": rec}
