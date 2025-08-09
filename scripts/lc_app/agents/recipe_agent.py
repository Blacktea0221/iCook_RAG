# agents/recipe_agent.py
import json
import logging
import os
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

# LLM 選擇：Ollama（預設）或 OpenAI（設 USE_OPENAI=true）
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-q4_K_M")

if USE_OPENAI:
    from langchain_openai import ChatOpenAI

    llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_presenter = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
else:
    from langchain_ollama import ChatOllama

    OLLAMA_URL = "http://127.0.0.1:11434"  # ★ 固定用本機，不吃 OLLAMA_HOST
    MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-q4_K_M")

    # Router：強制 JSON 模式，避免雜訊
    llm_router = ChatOllama(
        base_url=OLLAMA_URL, model=MODEL, temperature=0, format="json"
    )

    # Presenter：正常文字輸出，不要 JSON 模式
    llm_presenter = ChatOllama(base_url=OLLAMA_URL, model=MODEL, temperature=0.3)

from scripts.lc_app.tools.get_recipe_tool import get_recipe_tool
from scripts.lc_app.tools.recipe_search_tool import recipe_search_tool

TOOLS = [recipe_search_tool, get_recipe_tool]

# 簡易日誌（測試用）
logger = logging.getLogger("recipe_agent")
logger.setLevel(logging.INFO)


class Route(TypedDict):
    intent: Literal["recipe", "nutrition", "price", "identify", "other"]


BASE_DIR = Path(__file__).resolve().parent  # .../scripts/lc_app/agents
PROMPTS_DIR = BASE_DIR.parent / "prompts"  # .../scripts/lc_app/prompts

ROUTER_SYS = "你是嚴格的分類器，只能輸出 JSON。"
ROUTER_PROMPT = (PROMPTS_DIR / "router_prompt.txt").read_text(encoding="utf-8")
PRESENTER_SYS = (PROMPTS_DIR / "presenter_prompt.txt").read_text(encoding="utf-8")


def build_agent():
    def classify(text: str) -> dict:
        """Router：回傳 {"intent": "..."}；任何解析失敗都回 'other'，不讓 None 往下傳。"""
        # 1) 先試 structured output（OpenAI/支援良好時）
        try:
            router_chain = llm_router.with_structured_output(
                Route
            )  # Route 是 TypedDict
            data = router_chain.invoke(
                [
                    SystemMessage(content=ROUTER_SYS),
                    HumanMessage(content=ROUTER_PROMPT.replace("{text}", text)),
                ]
            )
            if isinstance(data, dict) and "intent" in data:
                data.setdefault("text", text)
                return data
        except Exception as e:
            logger.info(f"[Router] structured_output 失敗：{e}")

        # 2) 再試手動 JSON parse（Ollama 常見）
        try:
            res = llm_router.invoke(
                [
                    SystemMessage(content=ROUTER_SYS),
                    HumanMessage(content=ROUTER_PROMPT.replace("{text}", text)),
                ]
            )
            data = json.loads(res.content)
            # 成功路徑：
            if isinstance(data, dict) and "intent" in data:
                data["text"] = text
                return data
            # 失敗路徑：
            return {"intent": "other", "text": text}

        except Exception as e:
            logger.info(
                f"[Router] JSON 解析失敗：{e}, 原始回覆：{getattr(res, 'content', '')}"
            )

        # 3) 最後保險：不要讓 None 往下傳
        return {"intent": "other"}

    router = RunnableLambda(classify)

    # ---- 分支：只有 recipe 會呼叫 Tool ----
    def handle_recipe(route: dict):
        user_text = route.get("text", "")
        logger.info(f"[Router] intent=recipe, q={user_text}")
        # call tool
        results = recipe_search_tool.invoke(user_text)
        # 可選：如果使用者接著輸入「查看 12345」，可在 app.py 另開分支調 get_recipe_tool
        # 這裡先當「推薦清單呈現」
        # 直接沿用你的 llm_utils 讓 LLM 產生條列輸出（標題不得改寫，附 ID）
        from scripts.RAG.llm_utils import call_ollama_llm

        text = call_ollama_llm(user_text, results)
        return text

    def handle_nutrition(_: str):
        return "目前僅支援食譜查詢（食材/料理）。營養/熱量即將推出。"

    def handle_price(_: str):
        return "目前僅支援食譜查詢。價格/比價功能即將推出。"

    def handle_identify(_: str):
        return "目前僅支援食譜查詢。影像/辨識功能尚未啟用。"

    def handle_other(_: str):
        return "我可以幫你找食譜，請輸入你手邊的食材或想做的料理～"

    branch = RunnableBranch(
        (lambda x: x["intent"] == "recipe", RunnableLambda(handle_recipe)),
        (lambda x: x["intent"] == "nutrition", RunnableLambda(handle_nutrition)),
        (lambda x: x["intent"] == "price", RunnableLambda(handle_price)),
        (lambda x: x["intent"] == "identify", RunnableLambda(handle_identify)),
        RunnableLambda(handle_other),
    )

    # ---- 組成整條鏈：先路由，再分支 ----
    chain = router | branch
    return chain
