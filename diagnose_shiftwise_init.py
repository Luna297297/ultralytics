"""診斷 ShiftWise 初始化失敗的原因"""

import torch
import os

print("=" * 70)
print("ShiftWise 初始化診斷")
print("=" * 70)

# 1. 檢查 CUDA
print("\n1. CUDA 環境檢查")
print("-" * 70)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    try:
        torch.cuda.synchronize()
        print("✅ CUDA 同步成功")
    except Exception as e:
        print(f"❌ CUDA 同步失敗: {e}")

# 2. 檢查 shift-wiseConv import
print("\n2. ShiftWise 模組 Import 檢查")
print("-" * 70)
try:
    from ops.ops_py.add_shift import AddShift_mp_module
    print("✅ AddShift_mp_module import 成功")
    print(f"   模組: {AddShift_mp_module}")
    print(f"   位置: {AddShift_mp_module.__module__}")
except ImportError as e:
    print(f"❌ Import 失敗: {e}")
    print("   可能原因: shift-wiseConv 未正確安裝或編譯")
    exit(1)
except Exception as e:
    print(f"❌ 其他錯誤: {type(e).__name__}: {e}")
    exit(1)

# 3. 嘗試創建 AddShift_mp_module 實例
print("\n3. 嘗試創建 AddShift_mp_module 實例")
print("-" * 70)

# 測試參數
big_k = 13
small_k = 3
c_out = 32
import math
nk = math.ceil(big_k / small_k)
c_in = c_out * nk
group_in = 1

print(f"參數: big_k={big_k}, small_k={small_k}, c_out={c_out}, c_in={c_in}, nk={nk}")

try:
    # 啟用 CUDA_LAUNCH_BLOCKING 以獲取詳細錯誤
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    print("創建 AddShift_mp_module 實例...")
    shift_module = AddShift_mp_module(big_k, small_k, c_out, c_in, group_in)
    print("✅ AddShift_mp_module 創建成功")
    
    # 嘗試移到 CUDA
    if torch.cuda.is_available():
        print("將模組移到 CUDA...")
        shift_module = shift_module.cuda()
        print("✅ 模組已移到 CUDA")
    
    # 嘗試執行一次 forward（如果可能）
    if torch.cuda.is_available():
        print("\n4. 測試 Forward Pass")
        print("-" * 70)
        try:
            # 創建測試輸入
            b = 1
            h, w = 64, 64  # 小尺寸測試
            extra_pad = (small_k - 1) - small_k // 2
            hout = h - 2 * extra_pad
            wout = w - 2 * extra_pad
            
            test_input = torch.randn(b, c_in, h, w).cuda()
            print(f"測試輸入: {test_input.shape}")
            print(f"輸出尺寸: hout={hout}, wout={wout}")
            
            y1, y2, y3 = shift_module(test_input, b, hout, wout)
            print(f"✅ Forward pass 成功")
            print(f"   輸出形狀: y1={y1.shape}, y2={y2.shape}, y3={y3.shape}")
        except Exception as e:
            print(f"❌ Forward pass 失敗: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
except RuntimeError as e:
    error_msg = str(e)
    print(f"❌ RuntimeError: {error_msg}")
    if "CUDA" in error_msg or "cuda" in error_msg.lower() or "illegal" in error_msg.lower():
        print("\n這是一個 CUDA 錯誤，可能的原因：")
        print("1. CUDA kernel 編譯問題")
        print("2. CUDA/PyTorch 版本不相容")
        print("3. CUDA 上下文損壞")
        print("4. 記憶體問題")
        print("\n建議：")
        print("- 重新編譯 shift-wiseConv")
        print("- 檢查 CUDA 和 PyTorch 版本相容性")
        print("- 嘗試重啟 Colab runtime")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ 其他錯誤: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("診斷完成")
print("=" * 70)

