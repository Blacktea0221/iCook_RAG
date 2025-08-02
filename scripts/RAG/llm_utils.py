import json
import re
import subprocess
import textwrap

# ========== 食材抽取 LLM PROMPT ==========

LLM_PROMPT_INGREDIENT = """你是食材抽取助手，只回 JSON 陣列。從句子中找出食材名稱（只要名稱），依序輸出：
---
{text}
---"""


def call_llm_extract_ingredients(text, ingredient_set, model_name="qwen3:4b-q4_K_M"):
    """
    用 LLM 進行關鍵字食材抽取，回傳只在 ingredient_set 內的詞
    """
    prompt = LLM_PROMPT_INGREDIENT.format(text=text)
    res = subprocess.run(
        ["ollama", "run", model_name, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    try:
        items = json.loads(res)
    except json.JSONDecodeError:
        items = re.split(r"[，,]\s*", res)
    return [i.strip() for i in items if i.strip() in ingredient_set]


# ========== 智能推薦食譜 LLM PROMPT ==========


def call_ollama_llm(user_query, recipes, model="qwen3:4b-q4_K_M"):
    """
    用 LLM 幫使用者推薦料理（文字生成說明）
    recipes: list of dict (通常來自 search_engine.py 的檢索結果)
    """
    if not recipes:
        return "找不到符合的食譜。"

    # 組裝 context，每道菜名稱＋主要食材（只需重點資訊即可）
    context_blocks = []
    for r in recipes:
        rec = r["recipe"]
        ingredients_str = "、".join(
            i["ingredient_name"] for i in rec.get("ingredients", [])
        )
        context_blocks.append(
            f"【{rec.get('食譜名稱','')}】(ID: {r['id']})\n"
            f"主要食材：{ingredients_str}\n"
            f"簡要說明：可參考詳細步驟製作。"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = (
        f"以下是料理食譜的資訊：\n{context_text}\n\n"
        f"請扮演一位料理專家，根據這些食譜資訊，"
        f"用30字描述內容。\n"
        f"請在每道料理標題後標註其 ID（如：台式羅勒燒雞 (ID: 474705)），以便用戶後續查詢。\n"
        f"在最前面用1. 2. 3. 表示每道食譜的順序。\n"
        f"將所有食譜條列式分別描述內容。\n"
        f"回覆請直接進入主題，不需討論分析過程。\n"
        f"只能從下列提供的食譜中描述內容。\n"
        f"若發現內容重複，請合併為一條並只列一次。\n"
        f"請用繁體中文回答。"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Ollama 發生錯誤：{e.stderr.strip()}"


# ========== Google 搜尋摘要歸納 LLM PROMPT ==========


def summarize_search_results(user_query, search_results, model="qwen3:4b-q4_K_M"):
    """
    把多筆 Google 搜尋結果交給 LLM，讓他條列式總結
    search_results: list of dict (title, link, snippet)
    """
    blocks = []
    for r in search_results:
        blocks.append(f"【{r['title']}】\n{r['snippet']}\nLink: {r['link']}")
    context = "\n\n---\n\n".join(blocks)

    prompt = textwrap.dedent(
        f"""\n你將獲得來自 Google 搜尋「{user_query} 食譜」的結果摘要（如下 %%% 所示），請依據**僅提供的資訊**產出條列式清單。

✅ 每筆輸出請嚴格遵循以下格式（用全形逗號分隔）：
網頁標題，全形逗號，20 字左右的簡介，全形逗號，原始網址

⚠️ 請注意：
1. **只能基於提供的資訊內容回答，不得推論或自行補充**
2. 每則簡介**長度約為 20 字（18～22 字內）**
3. 結果以條列清單形式呈現，每筆結果獨立一行
4. 請全程使用**繁體中文**
5. 網址請保持原樣，不可修改或省略

%%%
{context}
%%%
"""
    )

    res = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return res.stdout.strip()


# ========== 你可以根據需求繼續擴充更多 LLM 互動函式 ==========
