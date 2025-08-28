# ingredient_utils.py
import json
import jieba
# from scripts.RAG.search_engine import fetch_all

def build_ingredient_set_from_json(json_path: str) -> set:
    """
    從 JSON 檔案建立 ingredient_set：讀取 ingredient 和 preview_tag 去重。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    words = set()
    for recipe_id, details in data.items():
        # 這裡根據你的 JSON 檔案結構，把相關詞彙加進來
        # 你的 JSON 檔看起來是 ID -> {vegetarian, uses_pork}
        # 如果你有食材或標籤，請調整程式碼以讀取它們

        # 這裡可以加入你 JSON 檔案中的其他關鍵字，讓 jieba 能夠辨識
        if details.get("vegetarian") is not None:
            words.add("素食")
            words.add("葷食")
        if details.get("uses_pork") is not None:
            words.add("豬肉")

    # 讓 jieba 能切出你的新詞彙
    for w in words:
        if w:
            jieba.add_word(w)
    print(f"[init] 食材字典已初始化（共 {len(words)} 項）")
    return words