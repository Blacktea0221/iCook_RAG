#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("測試 braintrust 導入...")

try:
    import braintrust

    print("✓ braintrust 主模組導入成功")
    print(f"braintrust 版本: {getattr(braintrust, '__version__', '未知')}")
except ImportError as e:
    print(f"✗ braintrust 主模組導入失敗: {e}")
    exit(1)

# 檢查 Evaluator 類別
try:
    from braintrust import Evaluator

    print("✓ Evaluator 類別導入成功")
    print(f"Evaluator 類別: {Evaluator}")
except ImportError as e:
    print(f"✗ Evaluator 類別導入失敗: {e}")

# 嘗試檢查是否需要 autoevals 套件
try:
    import autoevals

    print("✓ autoevals 套件已安裝")
    print("autoevals 中可用的評估器:")
    for attr in dir(autoevals):
        if not attr.startswith("_") and "eval" in attr.lower():
            print(f"  - {attr}")
except ImportError:
    print("✗ autoevals 套件未安裝")
    print("建議安裝: pip install autoevals")

# 測試創建自定義評估器
print("\n測試創建自定義評估器...")
try:

    def custom_qa_evaluator(output, expected=None, input=None):
        """自定義問答評估器"""
        if expected is None:
            return {"score": 0.5, "metadata": {"reason": "沒有預期答案"}}

        # 簡單的相似度檢查
        if expected.lower() in output.lower():
            return {"score": 1.0, "metadata": {"reason": "包含預期內容"}}
        else:
            return {"score": 0.0, "metadata": {"reason": "不包含預期內容"}}

    print("✓ 自定義評估器創建成功")

    # 測試評估器
    test_result = custom_qa_evaluator("這是一個素食食譜", "素食")
    print(f"測試結果: {test_result}")

except Exception as e:
    print(f"✗ 自定義評估器創建失敗: {e}")
