import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv

from scripts.langchain_project.agents.recipe_agent import build_agent
from scripts.database.ingredient_utils import build_ingredient_set_from_db

load_dotenv()

os.environ.setdefault("USE_OPENAI", "false")

logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_agent():
    """
    初始化並返回 LangChain Agent
    """
    build_ingredient_set_from_db()
    agent = build_agent()
    return agent


def main():
    agent = create_agent()

    print("RAG 智能代理（輸入 exit 離開）")
    # 修正：移除 Braintrust 相關的程式碼，改為單純的循環
    while True:
        q = input("\n請描述你有的食材或需求: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        try:
            ans = agent.invoke(q)
            output = str(ans)
            print("\n" + output)
        except Exception as e:
            error_message = f"代理程式執行錯誤：{str(e)}"
            print("\n" + error_message)


if __name__ == "__main__":
    main()
