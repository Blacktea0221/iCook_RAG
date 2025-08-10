import re

import jieba

RECIPE_KWS = ["食譜", "做法", "料理", "菜單", "煮", "炒", "燉", "烤", "推薦"]


def classify_intent(text: str) -> str:
    t = text.strip()
    if re.search(r"\b(id|ID)[:：]?\s*\d{4,}\b", t):
        return "recipe"
    if any(kw in t for kw in RECIPE_KWS):
        return "recipe"
    tokens = list(jieba.cut(t))
    if any(tok in ("食譜", "料理", "菜") for tok in tokens):
        return "recipe"
    return "other"
