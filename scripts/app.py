# app.py
import logging
import os

from dotenv import load_dotenv

from scripts.langchain_project.agents.recipe_agent import build_agent
from scripts.main import build_ingredient_set_from_db

load_dotenv()

# 建議：用環境變數切換推理端
# USE_OPENAI=true 時，請 export OPENAI_API_KEY
os.environ.setdefault("USE_OPENAI", "false")

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    # 1) 初始化 jieba 詞典（從 DB 抓 ingredient/preview_tag）
    build_ingredient_set_from_db()

    # 2) 啟動 agent
    agent = build_agent()

    print("RAG 智能代理（輸入 exit 離開）")
    while True:
        q = input("\n請描述你有的食材或需求: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        # 3) 跑一條鏈（自動路由 → 食譜工具 → 呈現）
        ans = agent.invoke(q)
        print("\n" + str(ans))

        # 4) 額外：使用者可輸入「查看ID」，你可以在這裡加 if 邏輯，呼叫 get_recipe_tool


if __name__ == "__main__":
    main()
