import json
import subprocess
import re
import pandas as pd
from tqdm import tqdm

# 1. 載入你的詳細食材 CSV
#    從專案根目錄打開 data/clean/...，並改用分號分隔
df = pd.read_csv(
    "data/raw/九層塔_食譜資料.csv", sep=";", engine="python", encoding="utf-8"
)


# 確認你的欄位名稱，就是剛剛 pandas 讀出的 columns：
#  ['id', '食譜名稱', '網址', '預覽食材', '詳細食材', '做法', '圖片相對路徑']
id_col = "id"
ingr_col = "詳細食材"

# 2. 定義系統提示（system prompt）
system_prompt = (
    "你是一個菜譜分類器，僅能二選一：素食 或 葷食。\n"
    "遇到這些蔬菜要判斷葷食:蔥 , 蒜 , 韭菜 , 洋蔥 "
    "請**只**回傳「素食」或「葷食」兩字，"
    "不要任何多餘文字、思考過程、解釋或引號。"
)
def extract_label(text: str) -> str:
    """
    从模型输出里抽取最后一个出现的“素食”或“葷食”。
    如果都没找到，就返回空串。
    """
    matches = re.findall(r"(素食|葷食)", text)
    return matches[-1] if matches else ""

def classify_with_ollama(recipe_id: str, ingredients: str) -> str:
    user_prompt = f"食譜 ID：{recipe_id}\n食材列表：{ingredients}"
    full_prompt = system_prompt + "\n\n" + user_prompt

    # 明确接管 stdout/stderr，并用 UTF-8 解码
    proc = subprocess.run(
        ["ollama", "run", "qwen3:4b-q4_K_M", full_prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="ignore",  # 或 "replace"
    )

    if proc.returncode != 0:
        print(f"[ERROR] ollama failed (code={proc.returncode}):")
        print(proc.stderr)
        return ""

    # 这里 proc.stdout 一定是 str，不会再是 None
    raw = proc.stdout.strip()
    label = extract_label(raw)
    if not label:
        print(f"[WARN] 无法从模型输出中找到标签，原始输出：{raw!r}")
    return label


# 3. 逐行執行分類並存回 DataFrame
results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="分類中"):
    ingredients = row[ingr_col]
    cat = classify_with_ollama(row[id_col], ingredients)
    # 調試用：印出 prompt 中的食材與模型回應，確保沒有空值
    print(f"ID={row[id_col]}  Ingredients={ingredients[:50]}...  ->  {cat}")
    results.append(cat)

df["素葷分類"] = results

# 4. 只輸出 id 與 素葷分類
out_df = df[[id_col, "素葷分類"]]

# 4a. 列印到終端，格式 “476284  葷”
for _, row in out_df.iterrows():
    print(f"{row[id_col]}  {row['素葷分類']}")

# 4b. （可選）存成純兩欄、無表頭、空格分隔的檔案
out_df.to_csv(
    "data/clean/九層塔/九層塔_id_and_classification.csv",
    index=False,
    header=False,  # 不輸出欄名
    sep=" ",
)
