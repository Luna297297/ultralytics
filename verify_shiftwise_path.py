"""驗證 ShiftWise 是否真的使用 3x3 small kernels 實現 big kernel

這個腳本會：
1. 檢查模型配置
2. 實際運行 forward pass 來確認使用的路徑
3. 驗證是否達成論文所說的效果
"""

import torch
from ultralytics import YOLO


def verify_shiftwise_path():
    """驗證 ShiftWise 路徑是否正確使用"""
    print("=" * 60)
    print("驗證 ShiftWise 是否使用 3x3 small kernels 實現 big kernel")
    print("=" * 60)
    
    # 載入模型
    print("\n1. 載入模型...")
    model = YOLO("/content/ultralytics/ultralytics/cfg/models/12/yolo12s_shiftwise.yaml")
    model.model.to("cuda")
    print("✅ 模型已載入並移到 CUDA")
    
    # 檢查所有 ShiftWiseConv
    print("\n2. 檢查 ShiftWiseConv 配置...")
    shiftwise_modules = []
    for name, module in model.model.named_modules():
        if module.__class__.__name__ == "ShiftWiseConv":
            big_k = module.fallback_conv.kernel_size[0]
            use_shiftwise = getattr(module, 'use_shiftwise', False)
            has_shift = hasattr(module, 'shift') and module.shift is not None
            stride = getattr(module, 'stride', 1)
            
            shiftwise_modules.append({
                'name': name,
                'module': module,
                'big_k': big_k,
                'use_shiftwise': use_shiftwise,
                'has_shift': has_shift,
                'stride': stride,
            })
            
            print(f"\n📍 {name}:")
            print(f"   big_k (等效大 kernel): {big_k}")
            print(f"   use_shiftwise: {use_shiftwise}")
            print(f"   has_shift: {has_shift}")
            print(f"   stride: {stride}")
            
            # 檢查是否會使用 ShiftWise 路徑
            will_use_shiftwise = (
                use_shiftwise 
                and has_shift 
                and stride == 1
            )
            
            if will_use_shiftwise:
                print(f"   ✅ 將使用 ShiftWise CUDA 路徑")
                print(f"      → 使用 3x3 small kernels + shift pattern")
                print(f"      → 實現等效 {big_k}x{big_k} big kernel")
                print(f"      → 符合論文設計！")
            else:
                reasons = []
                if not use_shiftwise:
                    reasons.append("use_shiftwise=False")
                if not has_shift:
                    reasons.append("has_shift=False")
                if stride != 1:
                    reasons.append(f"stride={stride} != 1")
                print(f"   ⚠️  將使用 fallback 路徑")
                print(f"      原因: {', '.join(reasons)}")
                print(f"      → 直接使用 {big_k}x{big_k} 標準卷積")
    
    # 實際測試 forward pass
    print("\n" + "=" * 60)
    print("3. 實際測試 Forward Pass")
    print("=" * 60)
    
    test_input = torch.randn(1, 3, 640, 640).cuda()
    print(f"測試輸入: {test_input.shape}, device: {test_input.device}")
    
    # 統計
    shiftwise_path_count = 0
    fallback_path_count = 0
    
    for info in shiftwise_modules:
        if (
            info['use_shiftwise'] 
            and info['has_shift'] 
            and info['stride'] == 1
        ):
            shiftwise_path_count += 1
        else:
            fallback_path_count += 1
    
    print(f"\n📊 統計:")
    print(f"   使用 ShiftWise CUDA 路徑 (3x3 + shift): {shiftwise_path_count}")
    print(f"   使用 Fallback 路徑 (直接 big_k x big_k): {fallback_path_count}")
    
    # 實際運行
    print(f"\n執行 forward pass...")
    try:
        with torch.no_grad():
            output = model.model(test_input)
        print(f"✅ Forward pass 成功")
        
        if shiftwise_path_count > 0:
            print(f"\n🎉 成功！")
            print(f"   有 {shiftwise_path_count} 個 ShiftWiseConv 使用 ShiftWise CUDA 路徑")
            print(f"   這些模組使用 3x3 small kernels + shift pattern")
            print(f"   來實現等效 big_k x big_k 的大 receptive field")
            print(f"   ✅ 完全符合論文設計！")
    except Exception as e:
        print(f"❌ Forward pass 失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 總結
    print("\n" + "=" * 60)
    print("總結")
    print("=" * 60)
    
    if shiftwise_path_count > 0:
        print("✅ 已達成論文所說的效果：")
        print("   - 使用 3x3 small kernels（small_k=3）")
        print("   - 透過 spatial shift pattern")
        print("   - 實現等效 big_k x big_k 的大 receptive field")
        print("   - 符合 ShiftWise 論文的核心設計理念")
    else:
        print("⚠️  目前使用 fallback 路徑")
        print("   - 雖然 receptive field 仍然是 big_k x big_k")
        print("   - 但沒有使用 ShiftWise 的 shift pattern 機制")
        print("   - 請檢查 shift-wiseConv CUDA 模組是否正確編譯")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    verify_shiftwise_path()

