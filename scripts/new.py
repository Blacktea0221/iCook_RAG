#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
new.py

根据最新文件结构和分类逻辑改写后的 search_and_retrieve_recipes 脚本：
- embeddings 元文件直接放在 data/embeddings/ 下，不再按蔬菜子文件夹组织
- 加载 index.json 用于未来加速（当前未使用）
- 加载 Meat and Vegetarian.json 进行 diet 与 uses_pork 过滤
- 主流程：通过子串检测分类槽位 → Jieba 分词抽取食材槽位 → 分类过滤 → 向量检索 → 结果过滤 → 智能推荐 & 详情查询
"""
import json
import os
import re
import subprocess
import textwrap
from collections import defaultdict
from typing import Dict, List

import jieba
import numpy as np
import pandas as pd
from googlesearch import search  # pip install googlesearch-python
from sentence_transformers import SentenceTransformer

# -------------------- 项目路径 --------------------
vege_name = "九層塔"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# -------------------- 文件路径 --------------------
# embeddings 相关文件直接放在 data/embeddings 下
tags_path = os.path.join(ROOT_DIR, "data", "embeddings", "tags.json")
embed_path = os.path.join(ROOT_DIR, "data", "embeddings", "embeddings.npy")
index_path = os.path.join(ROOT_DIR, "data", "embeddings", "index.json")
classify_path = os.path.join(ROOT_DIR, "data", "embeddings", "Meat and Vegetarian.json")

# 清洗数据仍按 vege_name 存储
cleaned_path = os.path.join(
    ROOT_DIR, "data", "clean", vege_name, f"{vege_name}_recipes_cleaned.csv"
)
preview_path = os.path.join(
    ROOT_DIR, "data", "clean", vege_name, f"{vege_name}_preview_ingredients.csv"
)
detailed_path = os.path.join(
    ROOT_DIR, "data", "clean", vege_name, f"{vege_name}_detailed_ingredients.csv"
)
steps_path = os.path.join(
    ROOT_DIR, "data", "clean", vege_name, f"{vege_name}_recipe_steps.csv"
)

# -------------------- 加载 embeddings 与模型 --------------------
with open(tags_path, "r", encoding="utf-8") as f:
    tags = json.load(f)
embeddings = np.load(embed_path)
model = SentenceTransformer("BAAI/bge-m3")
emb_norms = np.linalg.norm(embeddings, axis=1)

# 加载 index.json（未来可用于加速特定食材检索）
with open(index_path, "r", encoding="utf-8") as f:
    index_map = json.load(f)

# 加载 Meat and Vegetarian 分类文件
with open(classify_path, "r", encoding="utf-8") as f:
    classification = json.load(f)

# 建立 id -> set(tag)
id2tags = defaultdict(set)
for item in tags:
    rid = int(item["id"])
    id2tags[rid].add(item["tag"])

# -------------------- 加载清洗后的食谱数据 --------------------
print("载入清洗后的食谱数据...")
df_cleaned = pd.read_csv(cleaned_path, sep=";", encoding="utf-8-sig")
df_cleaned.columns = df_cleaned.columns.str.strip()
df_preview = pd.read_csv(preview_path, encoding="utf-8-sig").rename(columns=lambda x: x.strip())
df_detailed = pd.read_csv(detailed_path, encoding="utf-8-sig").rename(columns=lambda x: x.strip())
df_steps = pd.read_csv(steps_path, encoding="utf-8-sig").rename(columns=lambda x: x.strip())

# ==============================================================
# 构建食材字典 & Jieba 词典
# ==============================================================
def build_ingredient_set(df_preview: pd.DataFrame, df_detailed: pd.DataFrame) -> set:
    tags_set = set()
    for line in df_preview["preview_tag"]:
        tags_set.update(t.strip() for t in str(line).split(",") if t.strip())
    tags_set.update(df_detailed["ingredient_name"].astype(str).str.strip())
    return {t for t in tags_set if t and not re.fullmatch(r"\d+", t)}

ING_SET = build_ingredient_set(df_preview, df_detailed)
for w in ING_SET:
    jieba.add_word(w)
print(f"食材字典大小：{len(ING_SET)}")

# ==============================================================
# 分类槽位 & 映射
# ==============================================================
CLASS_DICT = {"素食", "葷食"}
CLASS_MAPPING = {"素食": "vegetarian", "葷食": "non_vegetarian"}

# ==============================================================
# 关键字抽取函数
# ==============================================================
LLM_PROMPT = textwrap.dedent(
    """
    你是食材抽取助手，只回 JSON 数组。请从句子中提取食材名称：
    ---
    {text}
    ---
    """
)

def jieba_extract(text: str) -> List[str]:
    clean = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    tokens = jieba.lcut(clean, cut_all=False)
    return [ing for ing in ING_SET if ing in text]

def llm_extract(text: str, model_name: str = "qwen3:4b-q4_K_M") -> List[str]:
    prompt = LLM_PROMPT.format(text=text)
    proc = subprocess.run([...],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore")
    proc = subprocess.run(["ollama","run",model_name,prompt],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore")
    raw = proc.stdout or ""
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        items = re.split(r"[，,]\s*", raw)
    return [i.strip() for i in items if i.strip() in ING_SET]

def pull_ingredients(user_text: str) -> List[str]:
    words = jieba_extract(user_text)
    return words if words else llm_extract(user_text)

# ==============================================================
# 向量检索 & 部分匹配
# ==============================================================
def search_by_partial_ingredients(query: str, top_k: int = 3):
    ingredients = [kw.strip() for kw in query.replace("，", ",").split(",") if kw.strip()]
    if not ingredients:
        return []
    id2count = {}
    for rid, tagset in id2tags.items():
        count = sum(any(kw in tag for tag in tagset) for kw in ingredients)
        if count>0:
            id2count[rid]=count
    if not id2count:
        return []
    q_emb = model.encode([query])[0]
    q_norm = np.linalg.norm(q_emb)
    sims = embeddings.dot(q_emb)/(emb_norms*q_norm+1e-10)
    id2score = {}
    for i,t in enumerate(tags):
        rid=int(t["id"])
        if rid in id2count:
            id2score[rid]=max(id2score.get(rid,float("-inf")),float(sims[i]))
    sorted_ids = sorted(id2count.keys(), key=lambda rid:(-id2count[rid],-id2score[rid]))[:top_k]
    results=[]
    for rid in sorted_ids:
        rec=get_recipe_by_id(rid)
        if rec:
            results.append({"id":rid,"score":id2score[rid],"recipe":rec})
    return results

# ==============================================================
# 获取单个食谱详情
# ==============================================================
def get_recipe_by_id(recipe_id: int)->Dict:
    rec=df_cleaned[df_cleaned["id"]==recipe_id]
    if rec.empty: return None
    rd=rec.iloc[0].to_dict()
    rd["preview_tags"]=df_preview[df_preview["id"]==recipe_id]["preview_tag"].tolist()
    det=df_detailed[df_detailed["id"]==recipe_id]
    rd["ingredients"]=det[["ingredient_name","quantity","unit"]].to_dict(orient="records")
    st=df_steps[df_steps["id"]==recipe_id].sort_values("step_no")
    rd["steps"]=st[["step_no","description"]].to_dict(orient="records")
    return rd

# ==============================================================
# 调用 Ollama 推荐
# ==============================================================
def call_ollama_llm(user_query:str,recipes:List[Dict],model_name:str="qwen3:4b-q4_K_M") ->str:
    if not recipes: return "找不到符合的食譜。"
    blocks=[]
    for r in recipes:
        rec=r["recipe"]
        ingr="、".join(i["ingredient_name"] for i in rec["ingredients"])
        blocks.append(f"【{rec['食譜名稱']}】(ID:{r['id']}) 主要食材：{ingr}")
    ctx="\n\n---\n\n".join(blocks)
    prompt=textwrap.dedent(f"""
        以下是料理資訊：
        {ctx}
        請扮演料理專家，用條列式推薦最適合 \"{user_query}\" 的食譜，每條30字以內，標註名稱與ID。
        用繁體中文。
    """
    )
    res=subprocess.run(["ollama","run",model_name,prompt],capture_output=True,text=True)
    return res.stdout.strip() if res.returncode==0 else f"Ollama錯誤: {res.stderr.strip()}"

# ==============================================================
# Google 后备
# ==============================================================
def google_search_recipes(keyword:str,k:int=5)->List[Dict]:
    query=f"{keyword} 食譜"
    return [{"title":item.title,"link":item.url,"snippet":item.description} for item in search(query,num_results=k,lang="zh-tw")]

def summarize_search_results(user_query:str,results:List[Dict],model_name:str="qwen3:4b-q4_K_M")->str:
    blocks=[f"【{r['title']}】\n{r['snippet']}\nLink:{r['link']}" for r in results]
    ctx="\n\n---\n\n".join(blocks)
    prompt=textwrap.dedent(f"""
        以下是Google搜索「{user_query}食譜」結果，請用條列格式輸出標題、20字內簡介、網址。
        用繁體中文。
        {ctx}
    """
    )
    res=subprocess.run(["ollama","run",model_name,prompt],capture_output=True,text=True)
    return res.stdout.strip()

# ==============================================================
# 可读输出
# ==============================================================
def pretty_print(item:Dict):
    rec=item['recipe']
    print(f"===ID{item['id']} 相似度{item['score']:.4f}===")
    print(f"名稱：{rec['食譜名稱']} 分類：{rec.get('vege_name','')}")
    print("──食材──")
    for i,ing in enumerate(rec['ingredients'],1): print(f"{i}.{ing['ingredient_name']}{ing['quantity']}{ing['unit']}")
    print("──步驟──")
    for s in rec['steps']: print(f"{s['step_no']}.{s['description']}")

# ==============================================================
# 主流程
# ==============================================================
if __name__=='__main__':
    print("RAG 智能推薦 (exit離開)")
    while True:
        raw=input("\n請描述你有的食材或需求: ").strip()
        if raw.lower() in ("exit","quit"): break

        # 分类检测
        classes=[cls for cls in CLASS_DICT if cls in raw]
        hates_pork="不吃豬肉" in raw
        allowed_ids=None
        if classes:
            diet=CLASS_MAPPING[classes[0]]
            allowed_ids={int(rid) for rid,v in classification.items() if v.get('diet')==diet}
            if hates_pork: allowed_ids={rid for rid in allowed_ids if not classification.get(str(rid),{}).get('uses_pork')}
        elif hates_pork:
            allowed_ids={int(rid) for rid,v in classification.items() if not v.get('uses_pork')}

        # 抽取关键词
        keywords=pull_ingredients(raw)
        if not keywords:
            if allowed_ids is not None:
                print(f"🔍 找到 {len(allowed_ids)} 道{classes[0] if classes else ''}食譜, 顯示前5筆：")
                valid=[]
                for rid in allowed_ids:
                    rec=get_recipe_by_id(rid)
                    if rec: valid.append((rid,rec))
                    if len(valid)>=5: break
                for rid,rec in valid:
                    print(f"- {rec.get('食譜名稱','')} (ID: {rid})")
                continue
            print("⚠️ 未檢測到食材，Google后备...")
            web=google_search_recipes(raw)
            if not web: print("🚫 无结果"); continue
            print(summarize_search_results(raw,web))
            continue

        # 本地检索
        query=", ".join(keywords)
        candidates=search_by_partial_ingredients(query)
        if allowed_ids is not None:
            candidates=[c for c in candidates if c['id'] in allowed_ids]
        if not candidates:
            print("⚠️ 本地无结果，Google后备...")
            web=google_search_recipes(query)
            if not web: print("🚫 无结果"); continue
            print(summarize_search_results(query,web))
            continue

        # 推荐与详情
        print("\n推荐：")
        print(call_ollama_llm(query,candidates))
        print("\n🔍 输入ID/名称查看更多; new重新; exit退出")
        id_map={c['recipe']['食譜名稱']:c['id'] for c in candidates}
        while True:
            sel=input("> ").strip()
            if sel.lower() in('exit','quit'): exit()
            if sel.lower()=='new': break
            rid=None
            if sel.isdigit(): rid=int(sel)
            else:
                for name,idx in id_map.items():
                    if sel in name: rid=idx; break
            if rid and any(c['id']==rid for c in candidates):
                pretty_print(next(c for c in candidates if c['id']==rid))
            else: print("无法识别")
