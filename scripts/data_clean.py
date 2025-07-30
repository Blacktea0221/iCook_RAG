import os
import re

import pandas as pd


# 1. 欄位名稱標準化：去除前後空白與全形空格
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.replace("　", "", regex=False)
    return df


# 2. 去除重複列：以 'id' 或 '網址' 作為主鍵
def drop_duplicates(df):
    key = "id" if "id" in df.columns else ("網址" if "網址" in df.columns else None)
    if key:
        df = df.drop_duplicates(subset=[key])
    return df


# 3. 文字欄位清理：前後去空白、合併多重空白
def clean_whitespace(df):
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)
    return df


# 4. 新增分類欄位：從檔名擷取食材名稱
def add_vege_name_column(df, filename):
    vege_name = os.path.basename(filename).split("_")[0]
    df["vege_name"] = vege_name
    return df


# 5. 處理預覽食材：切分並清洗標籤
def process_preview_list(df):
    if "預覽食材" in df.columns:
        df["preview_list"] = (
            df["預覽食材"]
            .str.split(r"[，,；;、/\\n]")
            .apply(
                lambda lst: [
                    re.sub(r"[^\w\u4e00-\u9fff]", "", tag.strip().lower())
                    for tag in lst
                    if tag.strip()
                ]
            )
        )
    return df


# 6. 輸出 Preview Ingredients 表格
def save_preview_table(df, vege_name, output_dir):
    if "preview_list" in df.columns:
        prev = (
            df[["id", "vege_name", "preview_list"]]
            .explode("preview_list")
            .rename(columns={"preview_list": "preview_tag"})
        )
        # 2b. 把「可略」一律清空、再去掉可能產生的空白
        unwanted = ["可略", "約80克"]
        # 先把所有要剔除的词都替换成空，再去空白
        prev["preview_tag"] = (
            prev["preview_tag"].replace("|".join(unwanted), "", regex=True).str.strip()
        )
        # 再删掉空串
        prev = prev[prev["preview_tag"] != ""]

        os.makedirs(output_dir, exist_ok=True)
        prev.to_csv(
            os.path.join(output_dir, f"{vege_name}_preview_ingredients.csv"),
            index=False,
            encoding="utf-8-sig",
        )


# 7. 處理詳細食材：切分、展平、解析數量與單位
def process_ingredients_table(df, vege_name, output_dir):
    if "詳細食材" not in df.columns:
        return

    # 1. 定義單位字典（含中英重量、體積、計數及常見中文量詞）+ # 長度由大到小：避免「小把」被當成「把」
    units_list = [
        "個人喜好",
        "一點點",
        "小把",
        "大把",
        "手把",
        "把",
        "片",
        "隻",
        "個",
        "根",
        "台",
        "盒",
        "碗",
        "朵",
        "塊",
        "張",
        "小份",
        "份",
        "條",
        "包",
        "粒",
        "葉",
        "匙",
        "大湯匙",
        "顆",
        "片",
        "株",
        "棵",
        "支",
        "瓣",
        "cc",
        "g",
        "克",
        "公克",
        "公斤",
        "kg",
        "斤",
        "ml",
        "毫升",
        "l",
        "公升",
        "茶匙",
        "小匙",
        "湯匙",
        "大匙",
        "杯",
        "小撮",
        "全下",
        "小罐",
        "罐",
        "大勺",
        "酌量",
    ]
    units = "|".join(units_list)
    num_pattern = r"\d+/\d+|\d+-\d+|\d+\.?\d*"

    # 2. 切分原始食材字串
    df["ingredients_list"] = df["詳細食材"].str.split(r"[，,；;、\n]")
    ing = (
        df[["id", "ingredients_list"]]
        .explode("ingredients_list")
        .rename(columns={"ingredients_list": "ingredient"})
    )
    ing["ingredient"] = ing["ingredient"].str.strip()

    # 3. 標準化符號並移除括號內註解
    ing["cleaned"] = (
        ing["ingredient"]
        .str.replace("ｇ", "g", regex=False)
        .str.replace("Ｇ", "g", regex=False)
        .str.replace("／", "/", regex=False)
        .str.replace("～", "-", regex=False)
        .str.replace(r"[()（）]", "", regex=True)
        .str.strip()
    )

    # 4. 解析數量與單位
    def parse_qty_unit(text):
        # 優先「適量」「少許」
        m_approx = re.search(r"(適量|少許)", text)
        if m_approx:
            return None, m_approx.group(1)

        # 中文數字+量詞（如「一把」「一小撮」）→ 數值化    m_cn = re.search(rf"([一二三四五六七八九十兩]+)\s*(?:{units})", text)
        m_cn = re.search(r"([一二三四五六七八九十兩半]+)\s*(" + units + r")", text)
        if m_cn:
            cn_map = {
                "一": 1,
                "二": 2,
                "三": 3,
                "四": 4,
                "五": 5,
                "六": 6,
                "七": 7,
                "八": 8,
                "九": 9,
                "十": 10,
                "兩": 2,
                "半": 0.5,
            }
            qty = cn_map.get(m_cn.group(1), None)
            return qty, m_cn.group(2)

        # 數字（含分數、範圍、浮點）+單位
        m = re.search(rf"({num_pattern})\s*({units})", text)
        if m:
            num_str, unit = m.group(1), m.group(2)
            if "/" in num_str:
                n, d = num_str.split("/")
                qty = float(n) / float(d) if d != "0" else None
            elif "-" in num_str:
                # 對於範圍（如 "1-2"），不做平均，直接不回傳數值
                return num_str, unit
            else:
                qty = float(num_str) if "." in num_str else int(num_str)
            return qty, unit

        # 新增：尾端數字當作 count
        m_trail = re.search(r"(\d+)$", text)
        if m_trail:
            return int(m_trail.group(1)), "count"

        # 純量詞，例如「小把」→ unit="小把", quantity=None
        m_only = re.search(rf"(?:{units})$", text)
        if m_only and not re.search(num_pattern, text):
            return None, m_only.group(0)

        # 純數字 fallback → count
        m2 = re.fullmatch(num_pattern, text)
        if m2:
            val = m2.group(0)
            qty = float(val) if "." in val else int(val)
            return qty, "count"

        return None, None

    qty_unit = ing["cleaned"].apply(parse_qty_unit)
    ing[["quantity", "unit"]] = pd.DataFrame(qty_unit.tolist(), index=ing.index)

    # 5. 取出純食材名稱（移除數字+單位、量詞、適量/少許、符號）rm1 = rf"({num_pattern})\s*({units})"
    cn_nums = "[一二三四五六七八九十兩半]+"
    rm1 = rf"(?:{cn_nums}|\d+/\d+|\d+-\d+|\d+\.?\d*)\s*({units})"

    rm2 = rf"(?:{units})$"
    ing["ingredient_name"] = (
        ing["cleaned"]
        .str.replace(rm1, "", regex=True)
        .str.replace(rm2, "", regex=True)
        .str.replace(r"(?:適量|少許)$", "", regex=True)
        .str.replace("可略", "", regex=False)  # 刪掉“可略”
        .str.replace("約", "", regex=False)
        .str.replace(r"[（）]", "", regex=True)  # 刪掉全形括號符號
        .str.strip()
    )

    # 6. 去除同 id+名稱+單位的重複行
    result = ing[["id", "ingredient", "ingredient_name", "quantity", "unit"]]
    result = result.drop_duplicates(subset=["id", "ingredient_name", "unit"])

    # 7. 輸出
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(
        os.path.join(output_dir, f"{vege_name}_detailed_ingredients.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# 8. 處理做法步驟：拆分、展平、去除編號符號、編號步驟
def process_steps_table(df, vege_name, output_dir):
    if "做法" in df.columns:
        df["steps_list"] = df["做法"].str.split(r"[。\.；;、/\\n]")
        steps = (
            df[["id", "steps_list"]]
            .explode("steps_list")
            .rename(columns={"steps_list": "description"})
        )
        # 去掉開頭/結尾多餘的數字或符號
        steps["description"] = (
            steps["description"]
            .str.replace(r"^[\d\W]+", "", regex=True)
            .str.replace(r"[\d\W]+$", "", regex=True)
        )
        steps = steps[steps["description"] != ""]
        steps["step_no"] = steps.groupby("id").cumcount() + 1

        # 儲存至 CSV
        os.makedirs(output_dir, exist_ok=True)
        steps.to_csv(
            os.path.join(output_dir, f"{vege_name}_recipe_steps.csv"),
            index=False,
            encoding="utf-8-sig",
        )


# 9. 輸出最終清理後的完整食譜，並刪除原始文字欄與中繼欄
def save_cleaned_recipes(df, vege_name, output_dir):
    # 刪掉原始「預覽食材」「詳細食材」「做法」以及中間產物 list 欄
    cols_to_drop = [
        "預覽食材",
        "詳細食材",
        "做法",
        "preview_list",
        "ingredients_list",
        "steps_list",
        "網址",
    ]
    # 1. 先把原本的 vege_name 从列列表里去除
    save_cols = [c for c in df.columns if c not in cols_to_drop + ["vege_name"]]

    # 2. 再在食譜名稱前插入 vege_name
    save_cols.insert(
        save_cols.index("食譜名稱"),
        "vege_name",
    )

    # 3. 按新顺序输出
    os.makedirs(output_dir, exist_ok=True)
    df[save_cols].to_csv(
        os.path.join(output_dir, f"{vege_name}_recipes_cleaned.csv"),
        index=False,
        encoding="utf-8-sig",
        sep=";",
    )


# 主流程
def main():
    input_dir = "test"  # 原始 CSV 檔案資料夾
    output_dir = "./data/clean"  # 清理後檔案輸出資料夾

    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(input_dir, filename)
        vege_name = os.path.splitext(filename)[0].split("_")[0]

        # 讀取原始資料
        df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")

        # 清理流程
        df = clean_column_names(df)
        df = drop_duplicates(df)
        df = clean_whitespace(df)
        df = add_vege_name_column(df, filename)
        df = process_preview_list(df)

        # 輸出各表        # 新增：每个类别一个子目录
        vege_name_dir = os.path.join(output_dir, vege_name)
        os.makedirs(vege_name_dir, exist_ok=True)

        save_preview_table(df, vege_name, vege_name_dir)
        process_ingredients_table(df, vege_name, vege_name_dir)
        process_steps_table(df, vege_name, vege_name_dir)
        save_cleaned_recipes(df, vege_name, vege_name_dir)

    print("資料清整完成，結果已輸出至：", output_dir)


if __name__ == "__main__":
    main()
