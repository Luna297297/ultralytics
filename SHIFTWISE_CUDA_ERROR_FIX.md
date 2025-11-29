# ShiftWise CUDA 錯誤修復指南

## 問題描述

在訓練時可能遇到以下錯誤：
```
CUDA error: an illegal memory access was encountered
```

這通常是因為 ShiftWise CUDA kernel 在處理某些張量形狀或記憶體佈局時出現問題。

## 解決方案

### 方案 1: 使用 Fallback 模式（推薦）

如果 CUDA kernel 不穩定，可以強制使用 fallback 模式（使用標準 13x13 卷積）：

```python
import os
os.environ["SHIFTWISE_USE_FALLBACK"] = "1"

from ultralytics import YOLO
model = YOLO("yolo12s_shiftwise.yaml")
```

或者在 Colab 中：
```python
%env SHIFTWISE_USE_FALLBACK=1
from ultralytics import YOLO
model = YOLO("/content/ultralytics/ultralytics/cfg/models/12/yolo12s_shiftwise.yaml")
```

**注意**：Fallback 模式仍然使用 13x13 的大 kernel，只是不使用 ShiftWise 的 shift pattern 機制。Receptive field 仍然是大的，只是計算方式不同。

### 方案 2: 自動 Fallback（已實現）

程式碼已經實現了自動 fallback 機制。如果 CUDA kernel 出錯，會自動切換到 fallback 模式，並在第一次出錯時打印警告。

### 方案 3: 重新編譯 shift-wiseConv

如果問題持續，可以嘗試重新編譯 shift-wiseConv：

```bash
cd /content/shift-wiseConv
rm -rf build/ *.so
python setup.py build_ext --inplace
```

### 方案 4: 檢查 CUDA 版本相容性

確保 PyTorch 和 CUDA 版本相容：

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 技術細節

### 為什麼會出現 CUDA 錯誤？

1. **記憶體存取越界**：CUDA kernel 可能在某些張量形狀下計算錯誤的記憶體地址
2. **參數不匹配**：輸入張量的通道數或尺寸與 CUDA kernel 預期不符
3. **CUDA 版本不相容**：編譯的 CUDA kernel 與運行時的 CUDA 版本不匹配

### Fallback 模式的影響

- **Receptive field**：仍然是 `big_k x big_k`（例如 13x13）
- **計算方式**：使用標準卷積而非 ShiftWise 的 shift pattern
- **效能**：標準卷積可能比 ShiftWise 稍慢，但更穩定
- **效果**：對於 receptive field 來說，效果應該類似

## 建議

1. **開發階段**：使用 fallback 模式確保訓練穩定
2. **實驗階段**：如果 CUDA kernel 穩定，可以嘗試使用 ShiftWise 路徑
3. **生產環境**：根據穩定性選擇合適的模式

## 驗證

檢查當前使用的模式：

```python
for name, module in model.model.named_modules():
    if module.__class__.__name__ == "ShiftWiseConv":
        use_shiftwise = getattr(module, 'use_shiftwise', False)
        has_shift = hasattr(module, 'shift') and module.shift is not None
        print(f"{name}: use_shiftwise={use_shiftwise}, has_shift={has_shift}")
```

如果 `use_shiftwise=False` 或 `has_shift=False`，表示正在使用 fallback 模式。

