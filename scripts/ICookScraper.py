"""
iCook 食譜爬蟲（類別版本）── 將 A 的功能、輸出格式整合進 B
author : ChatGPT‑merged
date   : 2025‑07‑16
"""

import csv
import os
import random
import re
import time
from typing import Dict, List, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# --------- 工具函式（檔案合法化、網址清洗） ----------------
_filename_re = re.compile(r'[\\/:*?"<>|]')  # Windows / Unix 都禁用的字元
_size_flag_re = re.compile(r"/[wh]:\d+/")  # iCook 縮圖網址的寬高參數


def safe_name(text: str) -> str:
    """把檔名裡不允許的符號都換成底線"""
    return _filename_re.sub("_", text).strip()


def full_image_url(url: str) -> str:
    """去掉例如 /w:200/ 或 /h:300/ 的縮圖限制，拿到原圖"""
    return _size_flag_re.sub("/", url) if url else url


def load_keywords_from_txt(txt_path: str) -> list[str]:
    """
    讀取純文字檔，每行一個關鍵字，去掉空行與前後空白
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# --------- 主類別 ----------------------------------------------------------
class ICookRecipeScraper:
    def __init__(
        self,
        image_root: str = "images",
        headers: Dict[str, str] | None = None,
        base_url: str = "https://icook.tw",
    ):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            headers
            or {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0 Safari/537.36"
                )
            }
        )

        # 圖片總資料夾（依關鍵字再分子資料夾）
        self.image_root = image_root
        os.makedirs(self.image_root, exist_ok=True)

    # --------------------- 一、搜尋列表 ----------------------------------
    def search_recipes(
        self, keyword: str, max_count: int = 3, page_limit: int = 1
    ) -> List[Dict]:
        """回傳含 id/name/url/preview_ingredients/img_url 的 dict 清單"""

        results: List[Dict] = []

        for page in range(1, page_limit + 1):
            url = f"{self.base_url}/search/{keyword}?page={page}"
            res = self.session.get(url, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")

            for card in soup.select("a.browse-recipe-link"):
                recipe: Dict = {}

                name_tag = card.select_one("h2.browse-recipe-name")
                preview_tag = card.select_one("p.browse-recipe-content-ingredient")
                article = card.select_one("article.browse-recipe-card")
                img_tag = card.select_one("img.browse-recipe-cover-img")
                href = card.get("href")

                # 安全檢查
                if not (name_tag and href and article):
                    continue

                recipe["id"] = article.get("data-recipe-id", "")
                recipe["name"] = name_tag.get_text(strip=True)
                recipe["url"] = urljoin(self.base_url, href)
                recipe["preview_ingredients"] = (
                    preview_tag.get_text(strip=True).replace("食材：", "")
                    if preview_tag
                    else ""
                )

                raw_img = (
                    img_tag.get("data-src") or img_tag.get("src") if img_tag else ""
                )
                recipe["img_url"] = full_image_url(raw_img)

                results.append(recipe)
                if len(results) >= max_count:
                    return results

            # 頁數跑完或已達 max_count
            if len(results) >= max_count:
                break

            time.sleep(random.uniform(1, 2))  # 禮貌休息

        return results

    # --------------------- 二、抓取詳細食譜 --------------------------------
    def get_recipe_details(self, url: str) -> Tuple[List[str], List[str]]:
        """回傳 ingredients list, steps list"""
        res = self.session.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        ingredients = []
        for li in soup.select("li.ingredient"):
            name = li.select_one(".ingredient-name")
            unit = li.select_one(".ingredient-unit")
            if name:
                ingredients.append(
                    f"{name.get_text(strip=True)}{unit.get_text(strip=True) if unit else ''}"
                )

        steps = [
            p.get_text(strip=True)
            for p in soup.select("p.recipe-step-description-content")
            if p.get_text(strip=True)
        ]

        return ingredients, steps

    # --------------------- 三、下載圖片 ------------------------------------
    def download_image(
        self, img_url: str, recipe_id: str, recipe_name: str, keyword: str
    ) -> str | None:
        """下載圖片到 images/<keyword>/，成功回傳路徑，失敗回 None"""

        if not img_url or img_url.startswith("data:image"):
            print(f"⚠️  跳過無效圖片網址：{img_url}")
            return None

        try:
            res = self.session.get(img_url, timeout=10)
            res.raise_for_status()
            sub_dir = os.path.join(self.image_root, safe_name(keyword))
            os.makedirs(sub_dir, exist_ok=True)

            filename = f"{recipe_id}_{safe_name(recipe_name)}.jpg"
            path = os.path.join(sub_dir, filename)

            with open(path, "wb") as f:
                f.write(res.content)

            print(f"✅  圖片已下載：{path}")
            return path

        except Exception as e:
            print(f"⚠️  下載圖片失敗：{e}")
            return None

    # --------------------- 四、寫 CSV -------------------------------------
    @staticmethod
    def save_to_csv(
        data: List[Dict], keyword: str, csv_path: str | None = None
    ) -> None:
        """存半形分號分隔的 CSV，並固定放在 recipe/ 資料夾"""

        # ---------- (1) 決定路徑 ----------
        # 先確保 recipe/ 這個資料夾存在
        recipe_dir = os.path.join("data", "raw")
        os.makedirs(recipe_dir, exist_ok=True)

        # CSV 檔名維持原本規則
        filename = csv_path or f"{keyword}_食譜資料.csv"

        # 把資料夾 + 檔名組成完整路徑
        full_path = os.path.join(recipe_dir, filename)

        # ---------- (2) 寫檔 ----------
        with open(full_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(
                f,
                delimiter=";",  # 欄位仍用分號分隔
                quotechar='"',
                quoting=csv.QUOTE_ALL,
            )
            writer.writerow(
                [
                    "id",
                    "食譜名稱",
                    "網址",
                    "預覽食材",
                    "詳細食材",
                    "做法",
                    "圖片相對路徑",
                ]
            )
            for item in data:
                writer.writerow(
                    [
                        item.get("id", ""),
                        item.get("name", ""),
                        item.get("url", ""),
                        item.get("preview_ingredients", ""),
                        ", ".join(item.get("ingredients", [])),
                        " / ".join(item.get("steps", [])),
                        item.get("img_path", ""),
                    ]
                )

        print(f"\n✅  已儲存 {len(data)} 筆資料 → {full_path}")

    # --------------------- 五、整合流程（對外唯一入口） --------------------
    def run(
        self,
        keywords: List[str] | str,
        max_recipes: int = 3,
        page_limit: int = 1,
    ) -> None:
        """keywords 可給 '高麗菜' 或 ['高麗菜', '馬鈴薯']"""

        if isinstance(keywords, str):
            keywords = [keywords]

        for kw in keywords:
            print(f"\n🔍  開始搜尋：「{kw}」")
            recipes = self.search_recipes(
                kw, max_count=max_recipes, page_limit=page_limit
            )
            print(f"→ 共找到 {len(recipes)} 筆\n")

            collected = []
            for idx, r in enumerate(recipes, 1):
                print(f"🍽  第 {idx} 筆｜{r['name']}")

                # 1) 下載圖片
                img_path = self.download_image(r["img_url"], r["id"], r["name"], kw)
                r["img_path"] = img_path or ""

                # 2) 詳細內容
                try:
                    r["ingredients"], r["steps"] = self.get_recipe_details(r["url"])
                except Exception as e:
                    print(f"⚠️  抓詳細失敗：{e}")
                    r["ingredients"], r["steps"] = [], []

                collected.append(r)
                print("-" * 40)
                time.sleep(random.uniform(1, 2))

            # 3) 輸出 CSV
            self.save_to_csv(collected, kw)


# -------------------- 若直接執行此檔 -------------------------
if __name__ == "__main__":
    # --- 讀 TXT 取得所有蔬菜名稱 ---
    TXT_FILE = "data\蔬菜.txt"  # ← 若檔名或路徑不同，這裡改
    keywords = load_keywords_from_txt(TXT_FILE)

    # --- 建立爬蟲物件 ---
    scraper = ICookRecipeScraper(image_root="data/images")

    # --- 逐一爬取 ---
    # 傳 list 進去即可；程式會對每個關鍵字各自輸出 CSV 與圖片資料夾
    scraper.run(
        keywords=keywords,
        max_recipes=50,  # 每種蔬菜要抓幾篇食譜，按需調整
        page_limit=3,  # 搜尋結果翻幾頁
    )
