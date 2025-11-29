"""測試不同輸入尺寸，找出導致 CUDA kernel 錯誤的尺寸"""

import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ops.ops_py.add_shift import AddShift_mp_module

print("=" * 70)
print("測試 ShiftWise CUDA kernel 在不同輸入尺寸下的表現")
print("=" * 70)

big_k = 13
small_k = 3
c_out = 32
import math
nk = math.ceil(big_k / small_k)
c_in = c_out * nk
group_in = 1
extra_pad = (small_k - 1) - small_k // 2

print(f"參數: big_k={big_k}, small_k={small_k}, c_out={c_out}, c_in={c_in}, nk={nk}")
print(f"extra_pad: {extra_pad}")

# 創建模組
shift_module = AddShift_mp_module(big_k, small_k, c_out, c_in, group_in).cuda()

# 測試不同尺寸
test_sizes = [
    (1, 64, 64),   # 小尺寸
    (1, 128, 128), # 中等尺寸
    (1, 256, 256), # 較大尺寸
    (1, 320, 320), # YOLO 常用尺寸
    (1, 640, 640), # YOLO 訓練尺寸
    (16, 64, 64),  # 小尺寸 + 批次
    (16, 128, 128),# 中等尺寸 + 批次
    (8, 320, 320), # YOLO 尺寸 + 批次
]

print("\n" + "=" * 70)
print("開始測試...")
print("=" * 70)

successful_sizes = []
failed_sizes = []

for b, h, w in test_sizes:
    hout = h - 2 * extra_pad
    wout = w - 2 * extra_pad
    
    if hout <= 0 or wout <= 0:
        print(f"⚠️  跳過 (b={b}, h={h}, w={w}): 輸出尺寸無效 (hout={hout}, wout={wout})")
        continue
    
    test_input = torch.randn(b, c_in, h, w).cuda()
    
    print(f"\n測試: batch={b}, h={h}, w={w}, hout={hout}, wout={wout}")
    print(f"  輸入形狀: {test_input.shape}")
    
    try:
        # 清除 CUDA 錯誤狀態
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        y1, y2, y3 = shift_module(test_input, b, hout, wout)
        
        # 檢查輸出
        torch.cuda.synchronize()
        result = y1 + y2 + y3
        
        print(f"  ✅ 成功")
        print(f"     輸出形狀: y1={y1.shape}, y2={y2.shape}, y3={y3.shape}")
        print(f"     result shape: {result.shape}")
        successful_sizes.append((b, h, w))
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ 失敗: {type(e).__name__}: {error_msg}")
        failed_sizes.append((b, h, w))
        
        # 嘗試恢復 CUDA 狀態
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass

print("\n" + "=" * 70)
print("測試結果總結")
print("=" * 70)
print(f"\n✅ 成功的尺寸 ({len(successful_sizes)} 個):")
for b, h, w in successful_sizes:
    print(f"   batch={b}, h={h}, w={w}")

print(f"\n❌ 失敗的尺寸 ({len(failed_sizes)} 個):")
for b, h, w in failed_sizes:
    print(f"   batch={b}, h={h}, w={w}")

if failed_sizes:
    print(f"\n⚠️  發現問題：某些尺寸會導致 CUDA kernel 錯誤")
    print(f"   這可能是 CUDA kernel 的 bug 或相容性問題")
    print(f"   建議：")
    print(f"   1. 重新編譯 shift-wiseConv")
    print(f"   2. 檢查 CUDA/PyTorch 版本相容性")
    print(f"   3. 暫時使用 fallback 模式（設置 SHIFTWISE_DISABLE=1）")
else:
    print(f"\n✅ 所有測試尺寸都成功！")

