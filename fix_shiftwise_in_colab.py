"""在 Colab 上修復 ShiftWise 模組載入問題

運行此腳本來：
1. 確保 shift-wiseConv 已正確編譯
2. 重新載入 ultralytics 模組
3. 重新載入模型，讓 ShiftWiseConv 使用動態檢查
"""

import sys
import importlib

print("=" * 60)
print("修復 ShiftWise 模組載入問題")
print("=" * 60)

# 步驟 1: 確認 shift-wiseConv 路徑
print("\n步驟 1: 檢查 shift-wiseConv 路徑...")
import os
shiftwise_path = "/content/shift-wiseConv"
if os.path.exists(shiftwise_path):
    if shiftwise_path not in sys.path:
        sys.path.insert(0, shiftwise_path)
        print(f"✅ 已將 {shiftwise_path} 加入 Python 路徑")
    else:
        print(f"✅ {shiftwise_path} 已在 Python 路徑中")
else:
    print(f"⚠️  {shiftwise_path} 不存在，請確認已正確 clone")

# 步驟 2: 嘗試 import ShiftWise 模組
print("\n步驟 2: 檢查 ShiftWise CUDA 模組...")
try:
    from ops.ops_py.add_shift import AddShift_mp_module
    print("✅ ShiftWise CUDA 模組載入成功")
    print(f"   模組: {AddShift_mp_module}")
    shiftwise_available = True
except Exception as e:
    print(f"❌ ShiftWise CUDA 模組載入失敗: {e}")
    print("\n請先編譯 shift-wiseConv:")
    print("  %cd /content/shift-wiseConv")
    print("  !python setup.py build_ext --inplace")
    shiftwise_available = False

# 步驟 3: 重新載入 ultralytics 模組
print("\n步驟 3: 重新載入 ultralytics 模組...")
try:
    # 重新載入 shiftwise 模組
    import ultralytics.nn.modules.shiftwise
    importlib.reload(ultralytics.nn.modules.shiftwise)
    print("✅ 已重新載入 ultralytics.nn.modules.shiftwise")
    
    # 重新載入 block 模組（因為它 import 了 shiftwise）
    import ultralytics.nn.modules.block
    importlib.reload(ultralytics.nn.modules.block)
    print("✅ 已重新載入 ultralytics.nn.modules.block")
    
    # 重新載入 tasks 模組
    import ultralytics.nn.tasks
    importlib.reload(ultralytics.nn.tasks)
    print("✅ 已重新載入 ultralytics.nn.tasks")
    
except Exception as e:
    print(f"⚠️  重新載入時出錯: {e}")
    print("   這可能不影響功能，請繼續下一步")

# 步驟 4: 重新載入模型
print("\n步驟 4: 重新載入模型...")
print("請運行以下程式碼來重新載入模型:")
print("""
from ultralytics import YOLO
model = YOLO("/content/ultralytics/ultralytics/cfg/models/12/yolo12s_shiftwise.yaml")

# 檢查 ShiftWiseConv 狀態
for name, module in model.model.named_modules():
    if module.__class__.__name__ == "ShiftWiseConv":
        big_k = module.fallback_conv.kernel_size[0]
        use_shiftwise = getattr(module, 'use_shiftwise', False)
        has_shift = hasattr(module, 'shift') and module.shift is not None
        
        print(f"📍 {name}:")
        print(f"   big_k: {big_k}")
        print(f"   use_shiftwise: {use_shiftwise}")
        print(f"   has_shift: {has_shift}")
        
        if use_shiftwise and has_shift:
            print(f"   ✅ 將使用 ShiftWise CUDA 路徑（3x3 + shift 實現等效 {big_k}x{big_k}）")
        else:
            print(f"   ⚠️  將使用 fallback（直接使用 {big_k}x{big_k} conv）")
""")

print("\n" + "=" * 60)
print("完成")
print("=" * 60)
if shiftwise_available:
    print("✅ ShiftWise CUDA 模組已可用")
    print("   重新載入模型後，ShiftWiseConv 應該會使用 ShiftWise CUDA 路徑")
else:
    print("⚠️  ShiftWise CUDA 模組不可用")
    print("   請先編譯 shift-wiseConv，然後重新運行此腳本")

