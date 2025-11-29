# Colab CUDA 版本不相容問題修復指南

## 問題診斷

### 版本對比

**shift-wiseConv README 要求：**
- CUDA 11.7
- PyTorch 1.10.0
- cudatoolkit 11.3

**Colab 實際環境：**
- CUDA 12.8 (PyTorch 使用)
- CUDA 12.5 (nvcc 編譯器)
- PyTorch 2.9.1

### 問題原因

CUDA kernel 是用 **CUDA 11.7** 編譯的，但 Colab 運行時是 **CUDA 12.8**。這會導致：
- CUDA illegal memory access 錯誤
- CUDA 上下文損壞
- 訓練無法繼續

## 解決方案：重新編譯 shift-wiseConv

### 步驟 1: 檢查當前 CUDA 版本

在 Colab 上運行：

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch CUDA 版本: {torch.version.cuda}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

!nvcc --version
```

### 步驟 2: 重新編譯 shift-wiseConv

```python
%cd /content/shift-wiseConv/shiftadd

# 清理舊的編譯文件
!rm -rf build/
!find . -name "*.so" -delete
!find . -name "*.o" -delete

# 重新編譯（使用當前 CUDA 版本）
!python setup.py build_ext --inplace

# 驗證編譯結果
!find . -name "*.so" | head -5
```

### 步驟 3: 測試編譯結果

```python
# 測試 import
try:
    from ops.ops_py.add_shift import AddShift_mp_module
    print("✅ Import 成功")
    
    # 測試創建實例
    import torch
    shift_module = AddShift_mp_module(13, 3, 32, 160, 1)
    shift_module = shift_module.cuda()
    print("✅ 模組創建成功")
    
    # 測試 forward
    test_input = torch.randn(1, 160, 64, 64).cuda()
    y1, y2, y3 = shift_module(test_input, 1, 62, 62)
    print(f"✅ Forward 成功: {y1.shape}, {y2.shape}, {y3.shape}")
    
except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
```

### 步驟 4: 重新載入 ultralytics

```python
# 重新 import ultralytics（使用新編譯的 CUDA kernel）
import importlib
import ultralytics.nn.modules.shiftwise
importlib.reload(ultralytics.nn.modules.shiftwise)

from ultralytics import YOLO
model = YOLO("/content/ultralytics/ultralytics/cfg/models/12/yolo12s_shiftwise.yaml")
model.model.to("cuda")

# 測試
test_input = torch.randn(1, 3, 640, 640).cuda()
with torch.no_grad():
    output = model.model(test_input)
print("✅ 模型載入和測試成功")
```

## 如果重新編譯仍然失敗

### 選項 1: 檢查編譯錯誤

```python
%cd /content/shift-wiseConv/shiftadd
!python setup.py build_ext --inplace 2>&1 | tee compile_log.txt
!cat compile_log.txt
```

### 選項 2: 檢查 CUDA 相容性

```python
import torch
print(f"PyTorch CUDA 版本: {torch.version.cuda}")
print(f"CUDA 計算能力: {torch.cuda.get_device_capability(0)}")

# 檢查是否有 CUDA 編譯器
import subprocess
result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
print(result.stdout)
```

### 選項 3: 暫時使用 Fallback（如果急需訓練）

```python
import os
os.environ["SHIFTWISE_DISABLE"] = "1"  # 完全禁用 ShiftWise

from ultralytics import YOLO
model = YOLO("yolo12s_shiftwise.yaml")
model.train(...)
```

## 預期結果

重新編譯後，應該：
1. ✅ Import 成功
2. ✅ 模組創建成功
3. ✅ Forward pass 成功
4. ✅ 訓練時不再出現 CUDA illegal memory access 錯誤

## 注意事項

1. **每次重啟 Colab Runtime 後都需要重新編譯**
   - Colab 的環境可能會重置
   - 建議將編譯步驟保存為 notebook cell

2. **編譯時間**
   - 重新編譯可能需要幾分鐘
   - 請耐心等待

3. **版本相容性**
   - CUDA 12.x 通常向後相容 CUDA 11.x 編譯的 kernel
   - 但某些情況下可能需要重新編譯

## 完整腳本（一鍵執行）

```python
# 1. 檢查版本
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

# 2. 重新編譯
%cd /content/shift-wiseConv/shiftadd
!rm -rf build/ && find . -name "*.so" -delete
!python setup.py build_ext --inplace

# 3. 測試
from ops.ops_py.add_shift import AddShift_mp_module
print("✅ 編譯成功！")

# 4. 重新載入模型
import importlib
import ultralytics.nn.modules.shiftwise
importlib.reload(ultralytics.nn.modules.shiftwise)

from ultralytics import YOLO
model = YOLO("/content/ultralytics/ultralytics/cfg/models/12/yolo12s_shiftwise.yaml")
print("✅ 模型載入成功！")
```

