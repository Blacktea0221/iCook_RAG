# scripts/langchain_project/agents/recipe_agent.py
import json
import logging
from pathlib import Path
from typing import Literal, TypedDict, List, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableBranch, RunnableLambda

from scripts.langchain_project.model import get_chat_model
from scripts.langchain_project.tools.get_recipe_tool import get_recipe_tool
from scripts.langchain_project.tools.recipe_search_tool import recipe_search_tool

load_dotenv()

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


# 依角色取得 LLM
llm_router = get_chat_model("router")       # 溫度 0，傾向穩定分類
llm_presenter = get_chat_model("presenter") # 溫度 0.3，較自然的輸出


def _format_recipes_for_presenter(user_text: str, recipes: List[Dict]) -> str:
    """
    將 RAG 的結果整理成 presenter prompt 的輸入內容。
    規則：標題請「精確照抄」recipe 名稱，並附 (ID: xxxx)。
    """
    if not recipes:
        return "（目前沒有任何食譜結果）"

    blocks = []
    for r in recipes:
        rec = r.get("recipe", {})
        title = (rec.get("recipe") or rec.get("食譜名稱") or "").strip()
        ingredients_str = "、".join(
            (i.get("ingredient") or i.get("ingredient_name") or "").strip()
            for i in rec.get("ingredients", [])
            if (i.get("ingredient") or i.get("ingredient_name"))
        )
        # 每筆內容盡量精簡，避免超長 prompt
        blocks.append(
            f"【{title}】 (ID: {r['id']})\n"
            f"主要食材：{ingredients_str}\n"
            f"說明：可參考詳細步驟製作。"
        )

    context_text = "\n\n---\n\n".join(blocks)

    # 這裡不硬寫規則，把規範交給 presenter_prompt.txt（你既有的模板）
    # 如果你希望更嚴格控制輸出格式，可把規則也放到 SystemMessage。
    user_msg = (
        f"以下是使用者查詢「{user_text}」取得的食譜清單，"
        f"每筆用【標題】與 ID 表示：\n\n{context_text}\n\n"
        f"請依系統提示產生條列輸出。"
    )
    return user_msg


def build_agent():
    def classify(text: str) -> dict:
        """Router：回傳 {"intent": "..."}；任何解析失敗都回 'other'，不讓 None 往下傳。"""
        # 1) 優先用 structured output（OpenAI 表現佳）
        try:
            router_chain = llm_router.with_structured_output(Route)  # Route 是 TypedDict
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

        # 2) 後備：手動 JSON parse
        try:
            res = llm_router.invoke(
                [
                    SystemMessage(content=ROUTER_SYS),
                    HumanMessage(content=ROUTER_PROMPT.replace("{text}", text)),
                ]
            )
            data = json.loads(res.content)
            if isinstance(data, dict) and "intent" in data:
                data["text"] = text
                return data
            return {"intent": "other", "text": text}
        except Exception as e:
            logger.info(f"[Router] JSON 解析失敗：{e}")

        return {"intent": "other", "text": text}

    router = RunnableLambda(classify)

    def handle_recipe(route: dict):
        user_text = route.get("text", "")
        logger.info(f"[Router] intent=recipe, q={user_text}")

        # 1) 先用 Tool 取 RAG 結果
        results = recipe_search_tool.invoke(user_text)

        # 2) 用 presenter LLM 生成條列輸出（OpenAI）
        if not results:
            return "找不到符合的食譜。"

        user_prompt = _format_recipes_for_presenter(user_text, results)
        res = llm_presenter.invoke(
            [SystemMessage(content=PRESENTER_SYS), HumanMessage(content=user_prompt)]
        )
        return getattr(res, "content", str(res))

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

    # 整條鏈：先路由，再分支
    chain = router | branch
    return chain
