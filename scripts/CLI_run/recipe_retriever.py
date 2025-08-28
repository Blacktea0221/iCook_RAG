import sys
import os
import json
from typing import List, Dict, Any

# 取得目前檔案的絕對路徑
current_file_path = os.path.abspath(__file__)
# 取得專案根目錄路徑
project_root_path = os.path.abspath(
    os.path.join(os.path.dirname(current_file_path), "../..")
)
# 將專案根目錄加入 Python 模組搜尋路徑
sys.path.insert(0, project_root_path)

# 從其他模組中匯入所需的函式
from scripts.RAG.llm_utils import call_openai_llm
from scripts.RAG.search_engine import get_recipe_by_id, fetch_all
from scripts.RAG.vectorstore_utils import embed_text_to_np

# 假設這裡匯入你的 reranker 模型或函式
# from sentence_transformers.cross_encoder import CrossEncoder
# reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# === 核心功能：4. recipe 相關檢索函式 ===
def recipe_retrieval(raw_input_text: str, top_k: int = 3) -> str:
    """
    執行 4. recipe 相關的檢索流程。
    這個流程專注於直接搜索食譜名稱或其描述。
    """
    print(f"處理使用者輸入: '{raw_input_text}'")

    # === 步驟一：取得使用者輸入的向量表示 ===
    try:
        query_embedding = embed_text_to_np(raw_input_text)
        if query_embedding.size == 0:
            return json.dumps(
                {"error": "無法取得使用者輸入的向量表示。"},
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        return json.dumps({"error": f"取得向量失敗: {e}"}, ensure_ascii=False, indent=2)

    # === 步驟二：第一階段 - 粗略檢索 (Retrieval) ===
    # 這裡我們設定一個較大的候選數量，讓 reranker 有更多資料可以篩選
    candidate_k = 50
    try:
        query_list = query_embedding.tolist()
        embedding_str = str(query_list).replace(" ", "")

        # 執行 SQL 查詢，取得候選食譜 ID 和其相似度分數
        sql = f"""
        SELECT
            id,
            recipe AS name,
            embedding <-> '{embedding_str}' AS similarity_score
        FROM main_recipe
        ORDER BY similarity_score ASC
        LIMIT {candidate_k};
        """
        rows = fetch_all(sql)
        if not rows:
            return json.dumps({"error": "查無結果。"}, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"向量搜索失敗: {e}"}, ensure_ascii=False, indent=2)

    # === 步驟三：整理候選食譜資料 ===
    candidate_results = []
    for r in rows:
        recipe_id = r.get("id")
        similarity_score = r.get("similarity_score")
        full_recipe_data = get_recipe_by_id(str(recipe_id))

        if full_recipe_data:
            candidate_results.append(
                {
                    "id": recipe_id,
                    "score": float(similarity_score),
                    "recipe": full_recipe_data,
                }
            )

    if not candidate_results:
        return json.dumps(
            {"error": "根據搜索 ID 找不到完整的食譜資料。"},
            ensure_ascii=False,
            indent=2,
        )

    # === 步驟四：第二階段 - 再排序 (Reranking) ===
    # 這個函式會將候選集重新排序，以找出最精確的結果
    re_ranked_results = rerank_results(raw_input_text, candidate_results)

    # === 步驟五：從再排序後的結果中，選取最終的 top_k ===
    final_results = re_ranked_results[:top_k]

    # === 步驟六：將結果傳給 LLM 進行整理與輸出 ===
    try:
        summary_str = call_openai_llm(raw_input_text, final_results)
        return summary_str
    except Exception as e:
        return json.dumps({"error": f"LLM 摘要失敗: {e}"}, ensure_ascii=False, indent=2)


# === 模擬 reranker 函式 ===
def rerank_results(user_query: str, results: List[Dict]) -> List[Dict]:
    """
    模擬 reranker 的功能。
    在實際應用中，你需要在這裡呼叫一個真正的 reranker 模型，
    根據使用者查詢和食譜內容，計算更精確的相關性分數。
    """
    print(f"正在對 {len(results)} 個候選食譜進行再排序...")

    # 模擬 reranker 的邏輯：
    # 這裡只是一個簡單的例子，假設我們基於某個條件進行再排序。
    # 實際的 reranker 會使用機器學習模型來計算分數。

    # 為了展示流程，我們這裡簡單地將原始結果反轉排序，以模擬"重新排序"
    # 你可以把這行註解掉，並換成你的 reranker 程式碼
    # results.reverse()

    # 備註：你應該在這裡使用 reranker 模型，
    # 範例： reranker_model.predict([(user_query, doc.get("recipe")) for doc in results])

    return sorted(results, key=lambda x: x["score"])


# === 測試區塊 ===
def test_recipe_retriever():
    """提供一些預設輸入進行測試"""
    print("--- 執行 4. 食譜檢索功能自動測試 ---")

    test_cases = [
        "想做一道麻婆豆腐",
        "有沒有簡單的家常蛋炒飯食譜",
        "蔥爆牛肉要怎麼做",
        "告訴我關於三杯雞的食譜",
    ]

    for case in test_cases:
        print(f"\n[測試案例] 輸入: '{case}'")
        summary_str = recipe_retrieval(case)

        try:
            summary_json = json.loads(summary_str)
            print(
                "\n[AI 推薦摘要]\n"
                + json.dumps(summary_json, ensure_ascii=False, indent=2)
            )
        except (json.JSONDecodeError, TypeError):
            print("\n[AI 推薦摘要]\n" + summary_str)

        print("-" * 50)


# 僅在直接執行此檔案時執行測試
if __name__ == "__main__":
    test_recipe_retriever()
