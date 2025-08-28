# agent_cli0828.py
import logging
import os
import sys

# 1. 匯入 load_dotenv 函式，讓程式知道它在哪
from dotenv import load_dotenv

# 2. 立即呼叫它，載入 .env 檔案中的變數
load_dotenv()
print(f"Braintrust API Key: {os.getenv('BRAINTRUST_API_KEY')}")

# 3. 接下來就可以匯入你的自定義模組了
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.langchain_project.agents.recipe_agent import build_agent
from scripts.database.ingredient_utils import build_ingredient_set_from_json

# 新增 Braintrust 的導入
import braintrust as bt

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_agent():
    """
    初始化並返回 LangChain Agent
    """
    json_file_path = "scripts/Processing data/processed_recipes_full.json"
    build_ingredient_set_from_json(json_file_path)
    agent = build_agent()
    return agent


# def main():
#     try:
#         # 將 bt.init 區塊移到這裡，確保整個程式執行都對應一個實驗
#         with bt.init(project="食譜代理專案") as experiment:
#             agent = create_agent()
    
#             print("RAG 智能代理（輸入 exit 離開）")
    
#             while True:
#                 q = input("\n請描述你有的食材或需求: ").strip()
#                 if q.lower() in ("exit", "quit"):
#                     break
    
#                 try:
#                     ans = agent.invoke(q)
#                     output = str(ans)
#                     print("\n" + output)
#                 except Exception as e:
#                     error_message = f"代理程式執行錯誤：{str(e)}"
#                     print("\n" + error_message)

#     except Exception as e:
#         # 捕捉 create_agent() 階段的錯誤
#         print(f"程式初始化失敗：{str(e)}")

def main():
    try:
        with bt.init(project="測試連線") as experiment:
            print("Braintrust 連線成功！")
            # 這裡可以寫入一個簡單的紀錄
            experiment.log(
                input={"query": "測試"},
                output={"response": "這是測試的回應"}
            )
            print("測試紀錄已上傳。")
    except Exception as e:
        print(f"Braintrust 連線失敗：{str(e)}")
        
if __name__ == "__main__":
    main()