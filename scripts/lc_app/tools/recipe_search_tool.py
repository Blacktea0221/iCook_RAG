import os

from dotenv import load_dotenv

load_dotenv()

# scripts/lc_app/tools/recipe_search_tool.py
from typing import Dict, List

from langchain_core.tools import tool

# 懶載入 ingredient_set：沿用你 main.py 的建立方式
from scripts.main import build_ingredient_set_from_db

# 核心檢索：你現有的 A 方案
from scripts.RAG.search_engine import pull_ingredients, tag_then_vector_rank

TOP_K = int(os.getenv("RAG_TOP_K", 5))

# 模組層快取，避免每次都重建
_ING_SET = None


def _get_ing_set():
    global _ING_SET
    if _ING_SET is None:
        _ING_SET = build_ingredient_set_from_db()  # 內部也會把詞加進 jieba 字典
    return _ING_SET


@tool("recipe_search", return_direct=False)
def recipe_search_tool(user_text: str) -> List[Dict]:
    """
    依使用者輸入（自然語句）進行食譜檢索。
    流程：Jieba 抽詞 -> OR 召回 -> 本機 bge-m3 cosine 排序 -> DB 補全。
    """
    ing_set = _get_ing_set()
    tokens = pull_ingredients(user_text, ing_set)
    results = tag_then_vector_rank(user_text, tokens_from_jieba=tokens, top_k=TOP_K)
    return results
