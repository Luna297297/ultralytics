"""調試 ShiftWise CUDA 錯誤的腳本

在 Colab 上運行此腳本來獲取詳細的錯誤信息
"""

import os
# 啟用 CUDA 同步模式以獲取詳細錯誤信息
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from ultralytics import YOLO

print("=" * 60)
print("調試 ShiftWise CUDA 錯誤")
print("=" * 60)
print(f"CUDA_LAUNCH_BLOCKING: {os.getenv('CUDA_LAUNCH_BLOCKING')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# 載入模型
print("\n載入模型...")
model = YOLO("/content/ultralytics/ultralytics/cfg/models/12/yolo12s_shiftwise.yaml")
model.model.to("cuda")

# 檢查 ShiftWiseConv 配置
print("\n檢查 ShiftWiseConv 配置...")
for name, module in model.model.named_modules():
    if module.__class__.__name__ == "ShiftWiseConv":
        print(f"\n📍 {name}:")
        print(f"   Input channels (c1): {module.channel_expand.in_channels if hasattr(module, 'channel_expand') and module.channel_expand else 'N/A'}")
        print(f"   Expanded channels: {module.channel_expand.out_channels if hasattr(module, 'channel_expand') and module.channel_expand else 'N/A'}")
        print(f"   Output channels (c2): {module.shift.c_out if hasattr(module, 'shift') and module.shift else 'N/A'}")
        print(f"   nk: {module.nk if hasattr(module, 'nk') else 'N/A'}")
        print(f"   use_shiftwise: {module.use_shiftwise}")
        print(f"   has_shift: {module.shift is not None}")
        print(f"   has_channel_expand: {module.channel_expand is not None if hasattr(module, 'channel_expand') else False}")
        break

# 創建測試輸入
print("\n創建測試輸入...")
test_input = torch.randn(1, 3, 640, 640).cuda()
print(f"Test input shape: {test_input.shape}")

# 嘗試運行 forward
print("\n嘗試運行 forward pass...")
try:
    with torch.no_grad():
        output = model.model(test_input)
    print("✅ Forward pass 成功")
except Exception as e:
    print(f"❌ Forward pass 失敗:")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")
    import traceback
    traceback.print_exc()

