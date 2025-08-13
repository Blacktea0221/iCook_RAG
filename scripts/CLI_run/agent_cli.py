import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv

from scripts.langchain_project.agents.recipe_agent import build_agent
from scripts.database.ingredient_utils import build_ingredient_set_from_db

load_dotenv()

# 建議：用環境變數切換推理端
# USE_OPENAI=true 時，請 export OPENAI_API_KEY
os.environ.setdefault("USE_OPENAI", "false")

logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_agent():
    """
    初始化並返回 LangChain Agent
    """
    # 初始化 jieba 詞典（從 DB 抓 ingredient/preview_tag）
    build_ingredient_set_from_db()

    # 啟動 agent
    agent = build_agent()
    return agent


def main():
    # 直接呼叫 create_agent，確保初始化流程一致
    agent = create_agent()

    print("RAG 智能代理（輸入 exit 離開）")
    while True:
        q = input("\n請描述你有的食材或需求: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        # 跑一條鏈（自動路由 → 食譜工具 → 呈現）
        ans = agent.invoke(q)
        print("\n" + str(ans))

        # 額外擴充可依照需求添加「查看ID」等指令


if __name__ == "__main__":
    main()
