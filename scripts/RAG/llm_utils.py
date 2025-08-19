import json
import os
import re
import textwrap
from typing import List, Dict, Any, Set

from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

# ========== 食材抽取 LLM PROMPT (使用OpenAI) ==========

LLM_PROMPT_INGREDIENT = """你是食材抽取助手，只回 JSON 陣列。從句子中找出食材名稱（只要名稱），依序輸出：
---
{text}
---"""


def call_llm_extract_ingredients(
    text: str, ingredient_set: set, model_name: str = None
) -> List[str]:
    """
    用 LLM 進行關鍵字食材抽取，回傳只在 ingredient_set 內的詞 (使用 OpenAI)
    """
    model = model_name or DEFAULT_OPENAI_MODEL
    prompt_text = LLM_PROMPT_INGREDIENT.format(text=text)

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一個食材抽取助手，只會回傳 JSON 陣列。",
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result_json = json.loads(response.choices[0].message.content)
        items = result_json.get("ingredients", [])
    except (json.JSONDecodeError, openai.APIError) as e:
        print(f"LLM抽取食材時發生錯誤: {e}")
        items = re.split(r"[，,]\s*", text)

    return [i.strip() for i in items if i.strip() in ingredient_set]


# ========== 智能推薦食譜 LLM PROMPT (使用OpenAI) ==========


def call_openai_llm(user_text: str, recipes: List[Dict], model: str = None) -> str:
    """
    用 LLM 幫使用者推薦料理（文字生成說明）(使用 OpenAI)
    recipes: list of dict (通常來自 search_engine.py 的檢索結果)
    回傳 JSON 字串
    """
    model = model or DEFAULT_OPENAI_MODEL

    if not recipes:
        # 如果沒有食譜，回傳一個包含錯誤訊息的 JSON
        return json.dumps({"error": "找不到符合的食譜。"}, ensure_ascii=False, indent=2)

    context_blocks = []
    for r in recipes:
        # 確保 r["recipe"] 存在
        rec = r.get("recipe", {})
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

    # 修正：更新 system_message 以要求 JSON 格式輸出
    system_message = (
        f'你是一位料理專家，請根據提供的料理食譜資訊（每道以【標題】與 ID 表示）來生成推薦。請嚴格以 JSON 格式回傳，格式為: `{{ "recommendations": [ ... ] }}`。\n'
        f'在 `recommendations` 陣列中，每筆物件包含 `"title"`、`"id"`、`"summary"` 三個鍵。\n\n'
        f"以下是料理食譜的資訊：\n{context_text}\n\n"
        f"請遵循以下規則來生成 JSON 內容：\n"
        f'1. **`"title"` 必須精確複製【】內文字，不可改寫。**\n'
        f'2. **`"id"` 必須精確複製食譜的 ID。**\n'
        f'3. **`"summary"` 用約 20～30 字總結主要做法或重點（不可捏造額外材料）。**\n'
        f"4. **只使用繁體中文。**\n"
        f"5. **如果內容類似，可以合併成一條推薦，但 `id` 欄位需以陣列形式包含所有相關 ID。**\n"
        f"6. **僅能使用提供的食譜與資訊，不能自行補充。**\n"
    )

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=500,
            # 修正：設定 response_format 為 JSON 物件
            response_format={"type": "json_object"},
        )
        # LLM 的回傳已經是 JSON 格式字串，直接回傳即可
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        print(f"OpenAI API 錯誤：{e}")
        return json.dumps(
            {"error": f"OpenAI API 錯誤：{str(e)}"}, ensure_ascii=False, indent=2
        )
    except Exception as e:
        print(f"其他錯誤：{e}")
        return json.dumps(
            {"error": f"發生未知錯誤：{str(e)}"}, ensure_ascii=False, indent=2
        )


# ========== Google 搜尋摘要歸納 LLM PROMPT (使用OpenAI) ==========


def summarize_search_results(
    user_query: str, search_results: List[Dict], model: str = None
) -> str:
    """
    把多筆 Google 搜尋結果交給 LLM，讓他條列式總結 (使用 OpenAI)
    search_results: list of dict (title, link, snippet)
    """
    model = model or DEFAULT_OPENAI_MODEL

    blocks = []
    for r in search_results:
        blocks.append(f"【{r['title']}】\n{r['snippet']}\nLink: {r['link']}")
    context = "\n\n---\n\n".join(blocks)

    system_message = textwrap.dedent(
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

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        print(f"OpenAI API 錯誤：{e}")
        return "很抱歉，在總結搜尋結果時發生錯誤。請稍後再試。"
    except Exception as e:
        print(f"其他錯誤：{e}")
        return "很抱歉，發生了未知的錯誤。"
