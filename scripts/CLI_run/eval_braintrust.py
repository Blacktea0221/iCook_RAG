import os
import sys
import braintrust
from dotenv import load_dotenv

# 加載環境變數
load_dotenv()

# 設定專案名稱
project_name = "iCook-RAG-Evaluation"

# 初始化 Braintrust（會自動從環境變數讀取 API Token）
braintrust.init(project=project_name)


# 定義自訂的問答評估器
def custom_qa_evaluator(output, expected=None, input=None):
    if expected is None:
        return {"score": 0.5, "metadata": {"reason": "沒有預期答案進行比較"}}
    output_lower = str(output).lower()
    expected_lower = str(expected).lower()
    if "素食" in expected_lower and "素食" in output_lower:
        return {"score": 1.0, "metadata": {"reason": "正確識別素食需求"}}
    elif "麻婆豆腐" in expected_lower and "麻婆豆腐" in output_lower:
        return {"score": 1.0, "metadata": {"reason": "正確回應麻婆豆腐查詢"}}
    elif any(keyword in output_lower for keyword in ["食譜", "製作", "步驟", "配料"]):
        return {"score": 0.8, "metadata": {"reason": "包含相關烹飪內容"}}
    else:
        return {"score": 0.2, "metadata": {"reason": "回應與預期不符"}}


# 定義測試案例
test_cases = [
    {
        "input": "請給我一個不含肉類的食譜",
        "expected": "找到的食譜應為素食，例如「香菇高麗菜飯」。",
        "metadata": {"test_case_id": "001"},
    },
    {
        "input": "如何製作麻婆豆腐？",
        "expected": "應包含麻婆豆腐的製作步驟和配料。",
        "metadata": {"test_case_id": "002"},
    },
]


# 定義代理任務
def run_agent_task(input_data):
    return "模擬的代理回應"


# 執行評估
braintrust.Eval(
    name="Agent Evaluation Run",
    data=test_cases,
    task=run_agent_task,
    scores=[custom_qa_evaluator],
)

print("評估任務已完成，請前往 Braintrust 儀表板查看結果。")
