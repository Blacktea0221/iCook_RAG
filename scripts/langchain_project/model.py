# scripts/langchain_project/model.py
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 統一從 .env 讀取設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 可選，預設官方端點

Role = Literal["router", "presenter", "default"]


def get_chat_model(
    role: Role = "default",
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """
    回傳配置好的 ChatOpenAI。
    role 可以快速帶入不同的溫度等預設：
      - router: 類分類/路由，較保守（溫度 0）
      - presenter: 摘要/重寫，略高（溫度 0.3）
      - default: 用 .env 預設
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY 未設定。請確認 .env 內已有 OPENAI_API_KEY。"
        )

    m = model or OPENAI_MODEL
    if role == "router":
        t = 0.0
        mt = max_tokens or 256
    elif role == "presenter":
        t = 0.3
        mt = max_tokens or OPENAI_MAX_TOKENS
    else:
        t = temperature if temperature is not None else OPENAI_TEMPERATURE
        mt = max_tokens or OPENAI_MAX_TOKENS

    # 依需求可加 base_url、timeout、重試等參數
    return ChatOpenAI(
        model=m,
        temperature=t,
        max_tokens=mt,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
    )
