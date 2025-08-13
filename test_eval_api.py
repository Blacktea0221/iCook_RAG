#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import braintrust
import inspect

print("檢查 braintrust.Eval 的參數...")

# 獲取 Eval 函數的簽名
sig = inspect.signature(braintrust.Eval)
print(f"Eval 函數簽名: {sig}")

print("\n參數詳情:")
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param.annotation} = {param.default}")

# 嘗試查看文檔字符串
print(f"\n文檔字符串:")
print(braintrust.Eval.__doc__)
