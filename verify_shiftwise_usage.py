"""驗證 ShiftWise 是否真的在使用，而不是 fallback 到標準卷積

這個腳本會詳細檢查：
1. ShiftWise 模組是否成功初始化
2. 是否真的使用 ShiftWise CUDA 路徑
3. 如何區分 ShiftWise 和 fallback
"""

import torch
from ultralytics import YOLO


def verify_shiftwise_usage():
    """詳細驗證 ShiftWise 的使用情況"""
    print("=" * 70)
    print("ShiftWise 使用情況驗證")
    print("=" * 70)
    
    # 載入模型
    print("\n1. 載入模型...")
    try:
        model = YOLO("/content/ultralytics/ultralytics/cfg/models/12/yolo12s_shiftwise.yaml")
        print("✅ 模型載入成功")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return
    
    # 移到 CUDA
    if torch.cuda.is_available():
        model.model.to("cuda")
        print("✅ 模型已移到 CUDA")
    else:
        print("⚠️  CUDA 不可用，ShiftWise 將使用 fallback")
    
    # 檢查所有 ShiftWiseConv
    print("\n" + "=" * 70)
    print("2. 檢查 ShiftWiseConv 模組狀態")
    print("=" * 70)
    
    shiftwise_modules = []
    for name, module in model.model.named_modules():
        if module.__class__.__name__ == "ShiftWiseConv":
            shiftwise_modules.append((name, module))
    
    if not shiftwise_modules:
        print("❌ 沒有找到 ShiftWiseConv 模組")
        return
    
    print(f"找到 {len(shiftwise_modules)} 個 ShiftWiseConv 模組\n")
    
    for idx, (name, module) in enumerate(shiftwise_modules, 1):
        print(f"📍 [{idx}] {name}:")
        print(f"   big_k (等效大 kernel): {module.fallback_conv.kernel_size[0]}")
        print(f"   use_shiftwise: {module.use_shiftwise}")
        print(f"   has_shift_module: {module.shift is not None}")
        print(f"   has_channel_expand: {module.channel_expand is not None if hasattr(module, 'channel_expand') else False}")
        
        if hasattr(module, '_shift_module_class') and module._shift_module_class is not None:
            print(f"   shift_module_class: {module._shift_module_class.__name__}")
            print(f"   ⚠️  尚未初始化（延遲初始化）")
        elif module.shift is not None:
            print(f"   ✅ ShiftWise 模組已初始化")
            print(f"      - c_out: {module.shift.c_out}")
            print(f"      - c_in: {module.shift.c_in}")
            print(f"      - nk: {module.nk}")
        else:
            print(f"   ❌ ShiftWise 模組未初始化")
        
        print()
    
    # 執行一次 forward 來觸發初始化
    print("=" * 70)
    print("3. 執行 Forward Pass 觸發初始化")
    print("=" * 70)
    
    test_input = torch.randn(1, 3, 640, 640)
    if torch.cuda.is_available():
        test_input = test_input.cuda()
    
    print(f"測試輸入: {test_input.shape}, device: {test_input.device}")
    
    try:
        with torch.no_grad():
            output = model.model(test_input)
        print("✅ Forward pass 成功")
    except Exception as e:
        print(f"❌ Forward pass 失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 再次檢查狀態（初始化後）
    print("\n" + "=" * 70)
    print("4. 檢查初始化後的狀態（Forward 後）")
    print("=" * 70)
    
    using_shiftwise_count = 0
    fallback_count = 0
    
    for idx, (name, module) in enumerate(shiftwise_modules, 1):
        print(f"\n[{idx}] {name}:")
        
        # 檢查是否成功初始化
        if module.shift is not None:
            print(f"   ✅ ShiftWise 模組已初始化")
            print(f"      - c_out: {module.shift.c_out}")
            print(f"      - c_in: {module.shift.c_in}")
            print(f"      - nk: {module.nk}")
        else:
            print(f"   ❌ ShiftWise 模組未初始化")
        
        # 檢查實際使用的路徑（通過 _path_used 標記）
        path_used = getattr(module, '_path_used', None)
        if path_used == 'shiftwise':
            print(f"   ✅✅✅ 實際使用 ShiftWise CUDA 路徑！")
            print(f"      → 使用 3x3 small kernels + shift pattern")
            print(f"      → 實現等效 {module.fallback_conv.kernel_size[0]}x{module.fallback_conv.kernel_size[0]} big kernel")
            print(f"      → 完全符合論文設計！")
            using_shiftwise_count += 1
        elif path_used == 'fallback':
            print(f"   ⚠️  ⚠️  ⚠️  實際使用 Fallback 路徑")
            print(f"      → 直接使用 {module.fallback_conv.kernel_size[0]}x{module.fallback_conv.kernel_size[0]} 標準卷積")
            print(f"      → 沒有使用 ShiftWise 的 shift pattern 機制")
            
            # 檢查 fallback 的原因
            reasons = []
            if not module.use_shiftwise:
                reasons.append("use_shiftwise=False")
            if module.shift is None:
                reasons.append("shift=None (初始化失敗)")
            if not hasattr(module, 'channel_expand') or module.channel_expand is None:
                reasons.append("channel_expand=None")
            if module.stride != 1:
                reasons.append(f"stride={module.stride} != 1")
            if not test_input.is_cuda:
                reasons.append("輸入不在 CUDA 上")
            
            if reasons:
                print(f"      原因: {', '.join(reasons)}")
            
            fallback_count += 1
        else:
            print(f"   ⚠️  尚未執行 forward，無法確定使用的路徑")
            print(f"      請執行一次 forward pass 來觸發初始化")
    
    # 總結
    print("\n" + "=" * 70)
    print("5. 總結")
    print("=" * 70)
    
    print(f"\n📊 統計:")
    print(f"   使用 ShiftWise CUDA 路徑: {using_shiftwise_count} 個模組")
    print(f"   使用 Fallback 路徑: {fallback_count} 個模組")
    
    if using_shiftwise_count > 0:
        print(f"\n✅ 成功！")
        print(f"   有 {using_shiftwise_count} 個 ShiftWiseConv 使用 ShiftWise CUDA 路徑")
        print(f"   這些模組使用 3x3 small kernels + shift pattern")
        print(f"   來實現等效 big_k x big_k 的大 receptive field")
        print(f"   ✅ 完全符合論文設計！")
    else:
        print(f"\n⚠️  警告：")
        print(f"   所有 ShiftWiseConv 都使用 fallback 路徑")
        print(f"   這表示 ShiftWise CUDA 模組初始化失敗或不可用")
        print(f"   請檢查：")
        print(f"   1. shift-wiseConv 是否正確編譯")
        print(f"   2. CUDA 是否可用")
        print(f"   3. PyTorch 和 CUDA 版本是否相容")
    
    # 提供驗證方法
    print("\n" + "=" * 70)
    print("6. 如何驗證實際運行時使用的路徑")
    print("=" * 70)
    
    print("""
方法 1: 檢查模組狀態（已執行）
  - 如果 shift is not None 且 use_shiftwise=True，會使用 ShiftWise 路徑

方法 2: 監控 forward 調用
  - 可以在 ShiftWiseConv.forward 中添加 print 語句
  - 查看是否進入 ShiftWise CUDA 路徑

方法 3: 比較計算結果
  - ShiftWise 路徑和 fallback 路徑的輸出應該不同
  - 可以保存兩種路徑的輸出進行比較

方法 4: 檢查 CUDA kernel 調用
  - 使用 nvidia-smi 或 CUDA profiler
  - 查看是否有 ShiftWise CUDA kernel 的調用
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    verify_shiftwise_usage()

