import re
import numpy as np
import jieba
import psycopg2

# from .vectorstore_utils import search_vectorstore
from .vectorstore_utils import embed_text_to_np

DB_CONFIG = dict(
    host="localhost", port=5432, database="postgres", user="lorraine", password="0000"
)


def fetch_one(query, params=None):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query, params or ())
    row = cur.fetchone()
    colnames = [desc[0] for desc in cur.description]
    conn.close()
    return dict(zip(colnames, row)) if row else None


def fetch_all(query, params=None):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query, params or ())
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    conn.close()
    return [dict(zip(colnames, row)) for row in rows]


# =============== 關鍵字抽取相關 ===============


def jieba_extract(text: str, ingredient_set: set) -> list:
    clean = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    tokens = jieba.lcut(clean, cut_all=False)
    return [t for t in tokens if t in ingredient_set]



def pull_ingredients(user_text: str, ingredient_set: set) -> list:
    """
    關鍵字抽取，預設只用 Jieba（不自動 fallback LLM）
    """
    words = jieba_extract(user_text, ingredient_set)
    return words


# =============== 食譜ID查詢 ===============
### 修改點：用 main_recipe / ingredient / recipe_steps 三表重寫
def get_recipe_by_id(recipe_id: int):
    rid = str(recipe_id)
    # 1) 主表：料理名稱（欄位名就叫 recipe）
    rec = fetch_one(
        "SELECT id, recipe, vege_id FROM main_recipe WHERE id = %s;",
        (rid,),
    )
    if not rec:
        return None

    # 2) 預覽 tag（ingredient 表的 preview_tag）
    tag_rows = fetch_all(
        "SELECT preview_tag FROM ingredient WHERE recipe_id = %s AND preview_tag IS NOT NULL;",
        (rid,),
    )
    preview_tags = [r["preview_tag"] for r in tag_rows]

    # 3) 食材明細（ingredient 表的 ingredient 欄位）
    ing_rows = fetch_all(
        "SELECT ingredient FROM ingredient WHERE recipe_id = %s;",
        (rid,),
    )
    # 這裡維持之前介面：recipe['ingredients'] 是 list[dict]
    ingredients = [{"ingredient": r["ingredient"]} for r in ing_rows]

    # 4) 步驟（recipe_steps：step_no、description）
    step_rows = fetch_all(
        "SELECT step_no, description FROM recipe_steps WHERE recipe_id = %s ORDER BY step_no;",
        (rid,),
    )
    steps = [{"step_no": r["step_no"], "description": r["description"]} for r in step_rows]

    # 5) 組合回傳結構（盡量沿用舊鍵名，避免其他程式碼要大改）
    recipe = {
        "id": rec["id"],
        "recipe": rec["recipe"],          # 料理名稱（舊程式若用 title，請同步調整）
        "vege_id": rec["vege_id"],        # 你資料裡有這欄，就保留
        "preview_tags": preview_tags,     # 來自 ingredient.preview_tag
        "ingredients": ingredients,       # 來自 ingredient.ingredient
        "steps": steps,                   # 來自 recipe_steps
    }
    return recipe



# =============== 檢索 ===============


### 修改點：工具函式—把 pgvector 的 text 轉回 numpy 向量
def _parse_pgvector_text(s: str) -> np.ndarray:
    """
    解析 SELECT embedding::text 取得的字串，例如 '[0.1, -0.2, ...]'
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return np.zeros((0,), dtype="float32")
    arr = np.fromstring(s, sep=",", dtype="float32")
    return arr

### 修改點：查 alias，回傳「輸入 tokens + 擴充別名」的集合
def _expand_aliases(tokens):
    if not tokens:
        return set()

    tokens_list = list(tokens)
    like_patterns = [f"%{t}%" for t in tokens_list]

    sql_eq = """
        SELECT DISTINCT alias
        FROM vege_alias
        WHERE alias = ANY(%s)
    """
    rows_eq = fetch_all(sql_eq, (tokens_list,))

    sql_like = """
        SELECT DISTINCT alias
        FROM vege_alias
        WHERE alias ILIKE ANY(%s)
    """
    rows_like = fetch_all(sql_like, (like_patterns,))

    expanded = set(tokens_list)
    expanded.update(r["alias"] for r in rows_eq)
    expanded.update(r["alias"] for r in rows_like)
    return expanded


### 修改點：第一階段 Tag 候選（OR）
def _candidate_recipe_ids_by_tag(expanded_terms, limit_per_term=200):
    """
    在 ingredient_vectors 表裡以 tag 等值/模糊匹配，回傳候選 recipe_id（去重）。
    預設用 OR（任何一個詞命中就收進候選）。
    """
    if not expanded_terms:
        return []

    # 用等值 + 模糊兩種，盡量撈到候選（避免 recall 太差）
    parts = []
    params = []

    parts.append("tag = ANY(%s)")
    params.append(list(expanded_terms))

    like_patterns = [f"%{t}%" for t in expanded_terms]
    parts.append("tag ILIKE ANY(%s)")
    params.append(like_patterns)

    sql = f"""
        SELECT DISTINCT recipe_id
        FROM ingredient_vectors
        WHERE ({' OR '.join(parts)})
        ORDER BY recipe_id
        LIMIT %s
    """

    params.append(limit_per_term * max(1, len(expanded_terms)))

    rows = fetch_all(sql, tuple(params))
    return [r["recipe_id"] for r in rows]

### 修改點：取出候選的所有 tag 向量（之後會用「整句向量」去比對）
def _fetch_tag_embeddings_for_recipes(recipe_ids):
    """
    回傳 [(recipe_id, tag, vege_name, embedding(np.ndarray)), ...]
    """
    if not recipe_ids:
        return []

    sql = """
        SELECT recipe_id, tag, vege_name, embedding::text AS embedding_text
        FROM ingredient_vectors
        WHERE recipe_id = ANY(%s)
    """
    rows = fetch_all(sql, (recipe_ids,))
    out = []
    for r in rows:
        vec = _parse_pgvector_text(r["embedding_text"])
        out.append((r["recipe_id"], r["tag"], r["vege_name"], vec))
    return out


### 修改點：主流程—先 Tag 縮小，再以「整句向量」對候選排序
def tag_then_vector_rank(
    user_text: str,
    tokens_from_jieba,    # 你抽到的關鍵詞列表
    top_k: int = 5
):
    """
    1) 擴充別名，先在 ingredient_vectors 用 tag 做 OR 篩選 -> 候選 recipe_id
    2) 將 user_text 轉向量
    3) 取出候選 recipe 的所有 tag 向量，分別與 user_text 向量做 cosine
       -> 對同一個 recipe 取「最大相似」作為該 recipe 的分數
    4) 按分數排序取前 top_k，並回傳完整食譜（get_recipe_by_id）
    """
    # 1) 擴充 tag
    expanded = _expand_aliases(tokens_from_jieba)
    # 如果 jieba 沒抓到，至少用原句裡所有非空白詞作為弱匹配（避免空集合）
    if not expanded:
        rough_tokens = [t for t in re.split(r"\s+|,|，|、|。", user_text) if t]
        expanded = _expand_aliases(rough_tokens)

    # 2) 先用 Tag OR 取候選
    candidate_ids = _candidate_recipe_ids_by_tag(expanded)
    if not candidate_ids:
        return []  # 完全沒命中，就讓外層走你的「全向量備援」或 Google 搜尋

    # 3) user_text -> 向量
    q = embed_text_to_np(user_text)
    q_norm = np.linalg.norm(q) + 1e-12

    # 4) 取候選的 tag 向量；以「recipe_id 最大相似分」做排序分數
    rows = _fetch_tag_embeddings_for_recipes(candidate_ids)

    best_score_by_id = {}
    tag_selected_by_id = {}
    vege_by_id = {}

    for rid, tag, vege_name, vec in rows:
        if vec.size == 0:
            continue
        v_norm = np.linalg.norm(vec) + 1e-12
        score = float(np.dot(q, vec) / (q_norm * v_norm))  # cosine
        if (rid not in best_score_by_id) or (score > best_score_by_id[rid]):
            best_score_by_id[rid] = score
            tag_selected_by_id[rid] = tag
            vege_by_id[rid] = vege_name

    # 5) 排序取前 K
    ranked = sorted(best_score_by_id.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 6) 補完整食譜內容
    results = []
    for rid, score in ranked:
        recipe = get_recipe_by_id(rid)
        if not recipe:
            continue
        results.append({
            "id": rid,
            "tag": tag_selected_by_id.get(rid),
            "vege_name": vege_by_id.get(rid),
            "score": score,
            "recipe": recipe,
        })
    return results
