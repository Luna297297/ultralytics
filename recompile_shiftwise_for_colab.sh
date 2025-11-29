#!/bin/bash
# 在 Colab 上重新編譯 shift-wiseConv 以匹配當前 CUDA 版本

echo "=========================================="
echo "重新編譯 shift-wiseConv 以匹配 Colab CUDA 版本"
echo "=========================================="

# 檢查 CUDA 版本
echo "1. 檢查 CUDA 版本..."
nvcc --version
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# 進入 shift-wiseConv 目錄
cd /content/shift-wiseConv/shiftadd

echo ""
echo "2. 清理舊的編譯文件..."
rm -rf build/
rm -f *.so
find . -name "*.so" -delete
find . -name "*.o" -delete

echo ""
echo "3. 重新編譯..."
python setup.py build_ext --inplace

echo ""
echo "4. 驗證編譯結果..."
if [ -f "ops/ops_py/add_shift.py" ]; then
    echo "✅ Python 模組存在"
else
    echo "❌ Python 模組不存在"
fi

# 檢查 .so 文件
SO_FILES=$(find . -name "*.so" | head -5)
if [ -n "$SO_FILES" ]; then
    echo "✅ 找到編譯的 .so 文件:"
    echo "$SO_FILES"
else
    echo "⚠️  未找到 .so 文件，可能需要檢查編譯錯誤"
fi

echo ""
echo "5. 測試 import..."
python -c "
try:
    from ops.ops_py.add_shift import AddShift_mp_module
    print('✅ Import 成功')
except Exception as e:
    print(f'❌ Import 失敗: {e}')
"

echo ""
echo "=========================================="
echo "編譯完成"
echo "=========================================="

