import json
import os
import re
import subprocess
import textwrap

from dotenv import load_dotenv

load_dotenv()

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-q4_K_M")

# ========== 食材抽取 LLM PROMPT ==========

LLM_PROMPT_INGREDIENT = """你是食材抽取助手，只回 JSON 陣列。從句子中找出食材名稱（只要名稱），依序輸出：
---
{text}
---"""


def call_llm_extract_ingredients(text, ingredient_set, model_name=None):
    model = model_name or DEFAULT_OLLAMA_MODEL
    """
    用 LLM 進行關鍵字食材抽取，回傳只在 ingredient_set 內的詞
    """
    prompt = LLM_PROMPT_INGREDIENT.format(text=text)
    res = subprocess.run(
        ["ollama", "run", model, prompt],
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


def call_ollama_llm(user_text, recipes, model=None):
    model = model or DEFAULT_OLLAMA_MODEL
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
        title = (rec.get("recipe") or rec.get("食譜名稱") or "").strip()
        ingredients_str = "、".join(
            (i.get("ingredient") or i.get("ingredient_name") or "").strip()
            for i in rec.get("ingredients", [])
            if (i.get("ingredient") or i.get("ingredient_name"))
        )
        context_blocks.append(
            f"【{title}】(ID: {r['id']})\n"
            f"主要食材：{ingredients_str}\n"
            f"簡要說明：可參考詳細步驟製作。"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = (
        f"以下是料理食譜的資訊（每道以【標題】與 ID 表示）：\n{context_text}\n\n"
        f"請扮演料理專家，依據以上**僅提供的資訊**，輸出條列清單：\n"
        f"1) 每條以阿拉伯數字編號\n"
        f"2) **標題請精確複製【】內文字，不可改寫或生成新標題**\n"
        f"3) 標題後標註其 ID（如： (ID: 474705) ）\n"
        f"4) 每條再用約 20～30 字摘要主要做法或重點（不可捏造額外材料）\n"
        f"5) 只使用繁體中文\n"
        f"6) 若內容類似可合併成一條，但保留所有涉及的 ID\n"
        f"7) 僅能使用提供的食譜與資訊，不能自行補充\n"
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
