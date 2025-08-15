import os
import sys
import json
import braintrust
from dotenv import load_dotenv

# Add the project root to the Python search path to resolve module imports.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../")
sys.path.insert(0, project_root)

# 加載環境變數
load_dotenv()

# 設定專案名稱
project_name = "iCook-RAG-Evaluation"

# 初始化 Braintrust（會自動從環境變數讀取 API Token）
braintrust.init(project=project_name)

# Import the agent creation function.
try:
    from scripts.CLI_run.agent_cli import create_agent

    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"警告：無法導入代理模組: {e}")
    print("將使用模擬回應進行測試")
    AGENT_AVAILABLE = False


# 定義自訂的問答評估器
def custom_qa_evaluator(output, expected=None, input=None):
    """
    評估代理回應的品質
    """
    try:
        # 增加嚴格的型態檢查，確保 output 是字串
        output = str(output) if output is not None else ""

        # 確保 expected 和 input 也是字串
        expected = str(expected) if expected is not None else ""
        input = str(input) if input is not None else ""

        if not expected:
            return braintrust.Score(
                name="qa_quality",
                score=0.5,
                metadata={"reason": "沒有預期答案進行比較"},
            )

        output_lower = output.lower()
        expected_lower = expected.lower()
        input_lower = input.lower()

        # 檢查是否有錯誤訊息，這是最高優先級的扣分項
        if "錯誤" in output_lower or "error" in output_lower:
            return braintrust.Score(
                name="qa_quality", score=0.1, metadata={"reason": "回應包含錯誤訊息"}
            )

        # 檢查特定關鍵詞匹配
        if "素食" in expected_lower and "素食" in output_lower:
            return braintrust.Score(
                name="qa_quality", score=1.0, metadata={"reason": "正確識別素食需求"}
            )
        elif "麻婆豆腐" in expected_lower and "麻婆豆腐" in output_lower:
            return braintrust.Score(
                name="qa_quality",
                score=1.0,
                metadata={"reason": "正確回應麻婆豆腐查詢"},
            )
        elif (
            "豬肉" in input_lower
            and "高麗菜" in input_lower
            and any(dish in output_lower for dish in ["炒高麗菜", "豬肉片", "高麗菜飯"])
        ):
            return braintrust.Score(
                name="qa_quality",
                score=1.0,
                metadata={"reason": "正確使用提供的食材推薦料理"},
            )
        elif "花生過敏" in input_lower and "花生" not in output_lower:
            return braintrust.Score(
                name="qa_quality", score=1.0, metadata={"reason": "正確避免過敏原"}
            )
        elif (
            "雞蛋" in input_lower and "豆腐" in input_lower and "花椰菜" in input_lower
        ):
            if any(
                dish in output_lower
                for dish in ["花椰菜炒蛋", "涼拌豆腐花椰菜", "蔬菜豆腐煲"]
            ):
                return braintrust.Score(
                    name="qa_quality",
                    score=1.0,
                    metadata={"reason": "正確使用素食食材"},
                )

        # 新增一個更通用的匹配邏輯，以防關鍵字匹配失敗
        # 檢查 output 是否包含 expected 中的任何關鍵字
        expected_keywords = expected_lower.split()
        if any(keyword in output_lower for keyword in expected_keywords):
            return braintrust.Score(
                name="qa_quality",
                score=0.8,
                metadata={"reason": "回應包含預期答案中的部分關鍵字"},
            )

        # 檢查是否包含基本烹飪相關內容
        cooking_keywords = ["食譜", "製作", "步驟", "配料", "料理", "烹煮", "食材"]
        if any(keyword in output_lower for keyword in cooking_keywords):
            return braintrust.Score(
                name="qa_quality", score=0.6, metadata={"reason": "包含相關烹飪內容"}
            )

        # 如果以上所有條件都不符合，給予一個基本分數
        return braintrust.Score(
            name="qa_quality", score=0.2, metadata={"reason": "回應與預期不符"}
        )

    except Exception as e:
        # 如果評估過程中發生任何錯誤，返回一個低分數
        return braintrust.Score(
            name="qa_quality", score=0.1, metadata={"reason": f"評估器錯誤: {str(e)}"}
        )


# 新增的函數：讀取並解析 few_shot_examples.txt
def load_test_cases_from_file(file_path):
    """
    從指定的檔案中讀取 JSON 格式的測試案例列表。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
            if not isinstance(test_cases, list):
                print(f"錯誤：檔案 {file_path} 的根物件不是一個列表。")
                return []
            return test_cases
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}，將使用空列表。")
        return []
    except json.JSONDecodeError as e:
        print(f"錯誤：解析檔案 {file_path} 時發生 JSON 錯誤: {e}")
        return []


# 定義 few-shot examples 檔案路徑（修正為絕對路徑）
few_shot_file_path = os.path.join(
    project_root, "scripts", "langchain_project", "prompts", "few_shot_examples.txt"
)

# 讀取測試案例
test_cases = load_test_cases_from_file(few_shot_file_path)

# 如果沒有讀取到任何測試案例，則發出警告
if not test_cases:
    print("警告：未讀取到任何測試案例，請檢查 few_shot_examples.txt 檔案。")


# 定義代理任務
def run_agent_task(input_data):
    """
    執行代理任務並記錄 prompt 到 Braintrust
    """
    # 開始一個新的 span 來記錄這次執行
    with braintrust.start_span(name="agent_task") as span:
        output = None  # 初始化 output 變數
        try:
            span.log(input=input_data)
            if AGENT_AVAILABLE:
                agent = create_agent()
                span.log(
                    metadata={
                        "agent_type": "iCook_RAG_Agent",
                        "model_used": os.getenv("OLLAMA_MODEL", "unknown"),
                        "use_openai": os.getenv("USE_OPENAI", "false"),
                    }
                )
                response = agent.invoke({"input": input_data})
                output = response.get("output", str(response))
                span.log(output=output)
                return output
            else:
                mock_response = f"模擬回應：針對「{input_data}」的食譜建議"
                span.log(output=mock_response)
                span.log(metadata={"mode": "mock"})
                return mock_response
        except Exception as e:
            # 在發生錯誤時，記錄錯誤訊息，並回傳一個包含錯誤訊息的字串
            error_message = f"代理程式執行錯誤：{str(e)}"
            span.log(error=error_message)
            span.log(metadata={"error_type": type(e).__name__})

            # 確保 output 變數有值，即使是錯誤訊息，這樣評估器也能處理
            output = error_message
            return output


# 執行評估
braintrust.Eval(
    name="Agent Evaluation Run",
    data=test_cases,
    task=run_agent_task,
    scores=[custom_qa_evaluator],
)

print("評估任務已完成，請前往 Braintrust 儀表板查看結果。")
