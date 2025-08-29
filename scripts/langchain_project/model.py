# scripts/langchain_project/model.py
"""
集中管理對話模型（LLM）的建立。
支援三種 profile：
- "router":     低溫度、短輸出、只做分類
- "sub_router": 低溫度、短輸出、做 recipe 子路由分類
- "presenter":  中低溫度、較自然的摘要輸出
- 其他值:       使用環境變數 OPENAI_* 的一般預設
"""
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 可選


def get_chat_model(
    profile: Optional[Literal["router", "sub_router", "presenter"]] = None,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """
    依 profile 回傳 ChatOpenAI。
    - 可使用 OPENAI_BASE_URL 指向兼容的供應商（如自架端點）
    """
    m = model or OPENAI_MODEL

    if profile in ("router", "sub_router"):
        # 盡量穩定、只做結構化輸出
        t = 0.0
        mt = max_tokens or 400
    elif profile == "presenter":
        # 摘要希望自然一些
        t = 0.3
        mt = max_tokens or 900
    else:
        t = OPENAI_TEMPERATURE if temperature is None else temperature
        mt = OPENAI_MAX_TOKENS if max_tokens is None else max_tokens

    return ChatOpenAI(
        model=m,
        temperature=t,
        max_tokens=mt,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
    )
