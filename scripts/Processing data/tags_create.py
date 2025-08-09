import glob
import json
import os
import re
import subprocess

import pandas as pd
from tqdm import tqdm

# 1. 載入你的詳細食材 CSV
#    從專案根目錄打開 data/clean/...，並改用分號分隔
RAW_DIR = "../../data/raw"
CSV_PATTERN = "*_食譜資料.csv"
all_dfs = []
for csv_path in glob.glob(os.path.join(RAW_DIR, CSV_PATTERN)):
    # 從檔名解析出 vege_name，例如 "九層塔"
    vege_name = os.path.basename(csv_path).split("_")[0]
    df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8")
    df["vege_name"] = vege_name  # 如果需要後面用到 vege_name
    all_dfs.append(df)

# 將所有菜種的 DataFrame 串成一個大表
df = pd.concat(all_dfs, ignore_index=True)


# 確認你的欄位名稱，就是剛剛 pandas 讀出的 columns：
id_col = "id"
ingr_col = "預覽食材"

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

result_dict = {}
results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="分類中"):
    rid = str(row[id_col])
    ingredients = row[ingr_col]
    label = classify_with_ollama(rid, ingredients)
    # optional: print(f"ID={rid} … ->  {label}")
    results.append(label)

    diet = "vegetarian" if label == "素食" else "non_vegetarian"
    uses_pork = bool(re.search(r"豬肉|豬排|五花肉|絞肉|排骨", row[ingr_col]))
    result_dict[rid] = {"diet": diet, "uses_pork": uses_pork}

df["素葷分類"] = results


# 确保 output 目录存在
out_dir = "../../data/embeddings"
os.makedirs(out_dir, exist_ok=True)

# 写入 Meat and Vegetarian.json
out_path = os.path.join(out_dir, "Meat and Vegetarian.json")
with open(out_path, "w", encoding="utf-8") as jf:
    json.dump(result_dict, jf, ensure_ascii=False, indent=2)

print(f"已儲存 Meat and Vegetarian.json 到：{out_path}")
