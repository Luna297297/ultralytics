"""ShiftWise convolution wrappers for YOLO modules."""

from __future__ import annotations

import os
import torch
import torch.nn as nn

# 嘗試載入 ShiftWise CUDA 模組（模組層級檢查）
try:
    from ops.ops_py.add_shift import AddShift_mp_module

    HAS_SHIFTWISE = True
except Exception:  # pragma: no cover - optional dependency
    AddShift_mp_module = None
    HAS_SHIFTWISE = False


def _check_shiftwise_available():
    """動態檢查 ShiftWise CUDA 模組是否可用（每次調用時重新檢查）"""
    try:
        from ops.ops_py.add_shift import AddShift_mp_module
        return True, AddShift_mp_module
    except Exception:
        return False, None


class ShiftWiseConv(nn.Module):
    """ShiftWise convolution module following the paper's design.
    
    This module implements large receptive field convolution using small 3x3 kernels
    with spatial shift patterns, as proposed in the ShiftWise paper.
    
    Args:
        c1: Input channels
        c2: Output channels
        big_k: Equivalent large kernel size (M in paper). Must be >> 3 (paper uses 13-51).
        small_k: Small kernel size, fixed to 3 per paper requirement.
        s: Stride (currently only stride=1 is supported)
        act: Activation function
    """

    def __init__(
        self, c1: int, c2: int, big_k: int = 13, small_k: int = 3, s: int = 1, act: bool | nn.Module = True
    ):
        super().__init__()
        self.stride = s
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        # Paper requirement: small_k must be fixed to 3
        if small_k != 3:
            raise ValueError(f"small_k must be 3 per paper requirement, got {small_k}")
        
        # Paper requirement: big_k must be >> 3 to achieve large receptive field
        if big_k <= 3:
            raise ValueError(f"big_k must be > 3 to achieve large receptive field effect, got {big_k}")

        # Fallback conv uses big_k for padding (to match receptive field size)
        padding = big_k // 2
        self.fallback_conv = nn.Conv2d(c1, c2, big_k, s, padding, bias=False)
        self.fallback_bn = nn.BatchNorm2d(c2)

        # 動態檢查 ShiftWise CUDA 模組是否可用（每次初始化時重新檢查）
        self.use_shiftwise, shift_module = _check_shiftwise_available()
        if self.use_shiftwise and shift_module is not None:
            self.shift = shift_module(big_k, small_k, c2, c1, group_in=1)
            self.shift_bn = nn.BatchNorm2d(c2)
        else:
            self.shift = None
            self.shift_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run shiftwise path when CUDA is available, otherwise fallback to standard conv."""
        # 檢查環境變數：如果設置了 SHIFTWISE_USE_FALLBACK=1，強制使用 fallback
        force_fallback = os.getenv("SHIFTWISE_USE_FALLBACK", "0") == "1"
        
        # 檢查是否可以使用 ShiftWise CUDA 路徑
        if (
            not force_fallback
            and self.use_shiftwise
            and self.shift is not None
            and x.is_cuda
            and self.stride == 1
        ):
            # 使用 ShiftWise CUDA 路徑（3x3 kernels + shift pattern 實現等效 big_k）
            b, c, h, w = x.shape
            
            # AddShift_mp_module.forward(x, b, hout, wout) 需要輸出尺寸
            # 內部計算：x_hin = hout + 2*extra_pad
            # 對於 small_k=3: extra_pad = (3-1) - 3//2 = 1
            # 所以如果輸入是 h，那麼 hout = h - 2*extra_pad = h - 2
            # 但實際上，由於我們使用 padding=big_k//2，輸出應該等於輸入
            # 為了匹配，我們需要傳遞 hout = h - 2*extra_pad
            try:
                # 計算 extra_pad（與 AddShift_mp_module 內部計算一致）
                # extra_pad = (small_k - 1) - small_k // 2
                small_k = 3  # 固定為 3
                extra_pad = (small_k - 1) - small_k // 2  # = 1
                
                # 計算輸出尺寸
                hout = h - 2 * extra_pad
                wout = w - 2 * extra_pad
                
                # 確保輸出尺寸是正數
                if hout <= 0 or wout <= 0:
                    raise ValueError(f"Invalid output size: hout={hout}, wout={wout} (input: h={h}, w={w})")
                
                # 確保張量是連續的（CUDA kernel 要求）
                if not x.is_contiguous():
                    x = x.contiguous()
                
                y1, y2, y3 = self.shift(x, b, hout, wout)
                
                # 確保輸出張量是連續的
                result = y1 + y2 + y3
                if not result.is_contiguous():
                    result = result.contiguous()
                
                return self.act(self.shift_bn(result))
            except (RuntimeError, ValueError) as e:
                # 如果 CUDA kernel 出錯或尺寸不匹配，fallback 到標準卷積
                error_msg = str(e)
                if "CUDA" in error_msg or "cuda" in error_msg.lower() or "illegal" in error_msg.lower():
                    # 只在第一次出錯時打印警告，避免刷屏
                    if not hasattr(self, '_fallback_warned'):
                        print(f"⚠️  ShiftWise CUDA kernel error, falling back to standard conv: {error_msg}")
                        self._fallback_warned = True
                    return self.act(self.fallback_bn(self.fallback_conv(x)))
                else:
                    raise
        else:
            # 使用 fallback 標準卷積（直接使用 big_k x big_k conv）
            return self.act(self.fallback_bn(self.fallback_conv(x)))

