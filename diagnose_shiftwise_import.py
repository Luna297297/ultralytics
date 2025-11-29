"""診斷 ShiftWise CUDA 模組載入問題"""

print("=" * 60)
print("診斷 ShiftWise CUDA 模組載入問題")
print("=" * 60)

# 1. 檢查 import 路徑
print("\n1. 檢查 import 路徑...")
try:
    import sys
    print(f"Python 路徑: {sys.path[:3]}...")  # 只顯示前3個
    
    # 檢查 ops 模組是否存在
    import os
    if '/content' in str(sys.path):
        print("檢查 /content 目錄...")
        if os.path.exists('/content/shift-wiseConv'):
            print("✅ /content/shift-wiseConv 存在")
            if os.path.exists('/content/shift-wiseConv/ops'):
                print("✅ /content/shift-wiseConv/ops 存在")
            else:
                print("❌ /content/shift-wiseConv/ops 不存在")
        else:
            print("❌ /content/shift-wiseConv 不存在")
except Exception as e:
    print(f"檢查路徑時出錯: {e}")

# 2. 嘗試直接 import
print("\n2. 嘗試直接 import AddShift_mp_module...")
try:
    from ops.ops_py.add_shift import AddShift_mp_module
    print("✅ 成功 import AddShift_mp_module")
    print(f"   模組類型: {type(AddShift_mp_module)}")
    print(f"   模組: {AddShift_mp_module}")
    HAS_SHIFTWISE = True
except ImportError as e:
    print(f"❌ ImportError: {e}")
    print("\n可能的原因:")
    print("  1. shift-wiseConv 沒有正確安裝")
    print("  2. CUDA 模組沒有編譯")
    print("  3. Python 路徑沒有包含 shift-wiseConv")
    HAS_SHIFTWISE = False
except Exception as e:
    print(f"❌ 其他錯誤: {type(e).__name__}: {e}")
    HAS_SHIFTWISE = False

# 3. 檢查 ultralytics 模組中的狀態
print("\n3. 檢查 ultralytics 模組中的 HAS_SHIFTWISE 狀態...")
try:
    from ultralytics.nn.modules.shiftwise import HAS_SHIFTWISE
    print(f"HAS_SHIFTWISE (在 ultralytics 中): {HAS_SHIFTWISE}")
    
    if not HAS_SHIFTWISE:
        print("\n⚠️  問題：ultralytics 模組載入時 ShiftWise CUDA 模組不可用")
        print("   這可能是因為:")
        print("   1. 在 import ultralytics 之前，shift-wiseConv 還沒有安裝/編譯")
        print("   2. 需要重新 import ultralytics 模組")
except Exception as e:
    print(f"檢查時出錯: {e}")

# 4. 提供解決方案
print("\n" + "=" * 60)
print("解決方案")
print("=" * 60)

if not HAS_SHIFTWISE:
    print("\n請按照以下步驟修復:")
    print("\n步驟 1: 確認 shift-wiseConv 已編譯")
    print("  %cd /content/shift-wiseConv")
    print("  !python setup.py build_ext --inplace")
    print("  或")
    print("  !pip install -e .")
    
    print("\n步驟 2: 確認 Python 路徑包含 shift-wiseConv")
    print("  import sys")
    print("  sys.path.insert(0, '/content/shift-wiseConv')")
    
    print("\n步驟 3: 重新載入 ultralytics 模組")
    print("  import importlib")
    print("  import ultralytics.nn.modules.shiftwise")
    print("  importlib.reload(ultralytics.nn.modules.shiftwise)")
    
    print("\n步驟 4: 重新載入模型")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('...')")
else:
    print("\n✅ ShiftWise CUDA 模組已成功載入")
    print("   如果模型仍然顯示 use_shiftwise=False，請重新載入模型")

