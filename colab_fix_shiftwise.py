"""
ShiftWise CUDA 版本修復 - Colab 一鍵執行腳本

根據 Gemini 分析，問題根源是：
1. PyTorch 版本差異太大（1.10.0 vs 2.9.1）
2. CUDA 版本差異（11.7 vs 12.8）
3. CUDA kernel 記憶體存取錯誤（需要重新編譯）

解決方案：
1. 重新編譯 shift-wiseConv（使用當前 CUDA 版本）
2. 確保所有張量記憶體連續（已加強在 shiftwise.py）
3. 如果失敗，考慮降級 PyTorch
"""

import os
import sys
import torch

print("=" * 70)
print("ShiftWise CUDA 版本修復腳本")
print("=" * 70)

# ============================================
print("\n[步驟 1] 檢查環境")
print("=" * 70)
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch CUDA 版本: {torch.version.cuda}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 計算能力: {torch.cuda.get_device_capability(0)}")
print(f"Python 版本: {sys.version}")

# 檢查 nvcc
print("\n檢查 nvcc 版本:")
os.system("nvcc --version 2>&1 | head -5")

# ============================================
print("\n[步驟 2] 重新編譯 shift-wiseConv")
print("=" * 70)

shiftwise_path = "/content/shift-wiseConv/shiftadd"
if not os.path.exists(shiftwise_path):
    print(f"❌ 找不到 {shiftwise_path}")
    print("請確保已 clone shift-wiseConv 到 /content/shift-wiseConv")
    sys.exit(1)

os.chdir(shiftwise_path)
print(f"當前目錄: {os.getcwd()}")

# 清理舊的編譯文件
print("\n清理舊的編譯文件...")
os.system("rm -rf build/ dist/ *.egg-info 2>/dev/null || true")
os.system("find . -name '*.so' -delete 2>/dev/null || true")
os.system("find . -name '*.o' -delete 2>/dev/null || true")
print("✅ 清理完成")

# 重新編譯
print("\n開始編譯 shift-wiseConv（使用當前 CUDA 版本）...")
compile_cmd = "python setup.py build_ext --inplace 2>&1 | tee compile_log.txt"
exit_code = os.system(compile_cmd)

# 檢查編譯結果
if os.path.exists("compile_log.txt"):
    with open("compile_log.txt", "r") as f:
        log = f.read()
        if "error" in log.lower() or "failed" in log.lower():
            print("⚠️  編譯可能有錯誤，請檢查 compile_log.txt")
            print("\n最後 50 行編譯日誌:")
            print("\n".join(log.split("\n")[-50:]))
        else:
            print("✅ 編譯完成（未發現明顯錯誤）")

# 檢查 .so 文件
so_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".so"):
            so_files.append(os.path.join(root, file))

if so_files:
    print(f"\n✅ 找到 {len(so_files)} 個編譯的 .so 文件:")
    for so_file in so_files[:5]:  # 只顯示前 5 個
        print(f"  - {so_file}")
else:
    print("\n⚠️  未找到 .so 文件，編譯可能失敗")

# ============================================
print("\n[步驟 3] 驗證編譯結果")
print("=" * 70)

try:
    # 添加路徑
    sys.path.insert(0, shiftwise_path)
    
    from ops.ops_py.add_shift import AddShift_mp_module
    print("✅ Import AddShift_mp_module 成功")
    
    # 測試創建模組
    print("\n測試創建模組...")
    shift_module = AddShift_mp_module(13, 3, 32, 160, 1)
    shift_module = shift_module.cuda()
    print("✅ 模組創建成功")
    
    # 測試 forward
    print("\n測試 forward pass...")
    test_input = torch.randn(1, 160, 64, 64).cuda()
    y1, y2, y3 = shift_module(test_input, 1, 62, 62)
    print(f"✅ Forward 測試成功")
    print(f"   輸出形狀: y1={y1.shape}, y2={y2.shape}, y3={y3.shape}")
    
    # 測試相加
    result = y1 + y2 + y3
    print(f"   相加結果: {result.shape}")
    print("✅ 所有測試通過")
    
except Exception as e:
    print(f"❌ 驗證失敗: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  如果驗證失敗，可能需要:")
    print("  1. 檢查編譯錯誤（compile_log.txt）")
    print("  2. 考慮降級 PyTorch（見 COMPREHENSIVE_FIX_GUIDE.md）")
    sys.exit(1)

# ============================================
print("\n[步驟 4] 重新載入 ultralytics")
print("=" * 70)

try:
    import importlib
    import ultralytics.nn.modules.shiftwise
    importlib.reload(ultralytics.nn.modules.shiftwise)
    print("✅ 重新載入 shiftwise 模組成功")
    
    from ultralytics import YOLO
    print("✅ Import YOLO 成功")
    
except Exception as e:
    print(f"❌ 重新載入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
print("\n[步驟 5] 測試模型")
print("=" * 70)

try:
    yaml_path = "/content/ultralytics/ultralytics/cfg/models/12/yolo12s_shiftwise.yaml"
    if not os.path.exists(yaml_path):
        print(f"⚠️  找不到 {yaml_path}，跳過模型測試")
    else:
        model = YOLO(yaml_path)
        model.model.to("cuda")
        print("✅ 模型載入成功")
        
        # 測試 forward
        print("\n測試模型 forward pass...")
        test_input = torch.randn(1, 3, 640, 640).cuda()
        with torch.no_grad():
            output = model.model(test_input)
        print(f"✅ 模型測試成功，輸出形狀: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        # 檢查使用的路徑
        print("\n檢查 ShiftWiseConv 模組狀態:")
        shiftwise_count = 0
        fallback_count = 0
        for name, module in model.model.named_modules():
            if module.__class__.__name__ == "ShiftWiseConv":
                path_used = getattr(module, '_path_used', None)
                if path_used == 'shiftwise':
                    shiftwise_count += 1
                elif path_used == 'fallback':
                    fallback_count += 1
        
        print(f"  - 使用 ShiftWise 路徑: {shiftwise_count} 個模組")
        print(f"  - 使用 Fallback 路徑: {fallback_count} 個模組")
        
        if shiftwise_count > 0:
            print("✅ 確認使用 ShiftWise CUDA 路徑")
        elif fallback_count > 0:
            print("⚠️  所有模組都使用 Fallback（標準卷積）")
        else:
            print("⚠️  未找到 ShiftWiseConv 模組")
            
except Exception as e:
    print(f"❌ 模型測試失敗: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  如果模型測試失敗，請檢查:")
    print("  1. YAML 配置文件是否正確")
    print("  2. 模型結構是否正確")
    print("  3. CUDA 記憶體是否足夠")

# ============================================
print("\n" + "=" * 70)
print("修復腳本執行完成")
print("=" * 70)
print("\n下一步:")
print("1. 如果所有測試通過，可以開始訓練")
print("2. 如果仍有問題，請查看 COMPREHENSIVE_FIX_GUIDE.md")
print("3. 每次重啟 Colab Runtime 後，需要重新執行此腳本")

