# scripts/langchain_project/tools/web_search_tool.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional

from googlesearch import search as google_search  # 非官方套件
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query_text: str = Field(..., description="要搜尋的查詢字串")
    top_k: int = Field(default=3, ge=1, le=10, description="回傳前幾筆結果")
    lang: str = Field(default="zh-tw", description="搜尋語言")
    pause: float = Field(default=2.0, description="查詢間隔（秒），避免被封鎖")
    timeout_sec: int = Field(default=10, description="整體搜尋逾時秒數")


class WebSearchOutput(BaseModel):
    status: Literal["ok", "empty", "error"]
    query: str
    results: List[Dict[str, str]] = Field(default_factory=list)  # [{title, link}]
    error: Optional[str] = None


class WebSearchTool:
    """
    輕量 Web 搜尋工具：
    - 使用 googlesearch-python（非官方）
    - 不下載頁面、不做 LLM
    - 輸出 [{title, link}]，最多 top_k 筆
    """

    def run(self, payload: dict) -> dict:
        args = WebSearchInput(**payload)
        start = time.time()
        raw_results, err = [], None
        try:
            want = max(args.top_k * 3, args.top_k)
            for item in google_search(
                args.query_text,
                advanced=True,
                num_results=want,
                lang=args.lang,
            ):
                if time.time() - start > args.timeout_sec:
                    err = "timeout"
                    break
                raw_results.append({"title": item.title, "link": item.url})
        except Exception as e:
            err = repr(e)

        if err:
            return WebSearchOutput(
                status="error", query=args.query_text, results=[], error=err
            ).model_dump()

        seen, uniq = set(), []
        for r in raw_results:
            link = r["link"]
            if link in seen:
                continue
            seen.add(link)
            uniq.append(r)
            if len(uniq) >= args.top_k:
                break

        if not uniq:
            return WebSearchOutput(
                status="empty", query=args.query_text, results=[], error=None
            ).model_dump()

        return WebSearchOutput(
            status="ok", query=args.query_text, results=uniq, error=None
        ).model_dump()
