# agents/recipe_agent.py
import json
import logging
import os
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

# --- LLM 設定 ---
# 統一使用 OpenAI，模型指定為 gpt-4.1-nano
llm_router = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
llm_presenter = ChatOpenAI(model="gpt-4.1-nano", temperature=0.3)

from scripts.langchain_project.tools.get_recipe_tool import get_recipe_tool
from scripts.langchain_project.tools.recipe_search_tool import recipe_search_tool

TOOLS = [recipe_search_tool, get_recipe_tool]

# 簡易日誌（測試用）
logger = logging.getLogger("recipe_agent")
logger.setLevel(logging.INFO)


class Route(TypedDict):
    intent: Literal["recipe", "nutrition", "price", "identify", "other"]


BASE_DIR = Path(__file__).resolve().parent  # .../scripts/langchain_project/agents
PROMPTS_DIR = BASE_DIR.parent / "prompts"  # .../scripts/langchain_project/prompts

ROUTER_SYS = "你是嚴格的分類器，只能輸出 JSON。"
ROUTER_PROMPT = (PROMPTS_DIR / "router_prompt.txt").read_text(encoding="utf-8")
PRESENTER_SYS = (PROMPTS_DIR / "presenter_prompt.txt").read_text(encoding="utf-8")


def build_agent():
    def classify(text: str) -> dict:
        """Router：回傳 {"intent": "..."}；任何解析失敗都回 'other'，不讓 None 往下傳。"""
        try:
            # 使用 with_structured_output 確保輸出格式正確
            router_chain = llm_router.with_structured_output(Route)
            data = router_chain.invoke(
                [
                    SystemMessage(content=ROUTER_SYS),
                    HumanMessage(content=ROUTER_PROMPT.replace("{text}", text)),
                ]
            )
            if isinstance(data, dict) and "intent" in data:
                data.setdefault("text", text)
                return data
            # 如果失敗，回傳 'other'
            return {"intent": "other", "text": text}
        except Exception as e:
            logger.info(f"[Router] structured_output 失敗：{e}")
            return {"intent": "other", "text": text}

    router = RunnableLambda(classify)

    # ---- 分支：只有 recipe 會呼叫 Tool ----
    def handle_recipe(route: dict):
        user_text = route.get("text", "")
        logger.info(f"[Router] intent=recipe, q={user_text}")
        results = recipe_search_tool.invoke(user_text)

        # 這裡改用 llm_presenter 進行摘要
        text = llm_presenter.invoke(
            [
                SystemMessage(content=PRESENTER_SYS),
                HumanMessage(
                    content=f"使用者查詢：{user_text}\n\n食譜資訊：{json.dumps(results, ensure_ascii=False, indent=2)}"
                )
            ]
        ).content
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