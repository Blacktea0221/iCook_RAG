import re

import jieba

from scripts.RAG.search_engine import fetch_all


def build_ingredient_set_from_db() -> set:
    """
    直接從資料庫建立 ingredient_set：抓 ingredient + preview_tag 去重。
    表：public.ingredient（recipe_id, ingredient, preview_tag）
    """
    rows = fetch_all(
        "SELECT ingredient, preview_tag FROM public.ingredient WHERE ingredient IS NOT NULL OR preview_tag IS NOT NULL;"
    )
    words = set()
    for r in rows:
        ing = (r.get("ingredient") or "").strip()
        tag = (r.get("preview_tag") or "").strip()
        if ing:
            words.add(ing)
        if tag:
            words.add(tag)
            for t in re.split(r"[ ,，、/|｜\+和]+", tag):
                t = t.strip()
                if t:
                    words.add(t)
    # 讓 jieba 能切出你資料庫裡的詞
    for w in words:
        if w:
            jieba.add_word(w)
    print(f"[init] 食材字典已初始化（共 {len(words)} 項）")
    return words
