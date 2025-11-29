"""ShiftWise convolution wrappers for YOLO modules."""

from __future__ import annotations

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
        # 檢查是否可以使用 ShiftWise CUDA 路徑
        if (
            self.use_shiftwise
            and self.shift is not None
            and x.is_cuda
            and self.stride == 1
        ):
            # 使用 ShiftWise CUDA 路徑（3x3 kernels + shift pattern 實現等效 big_k）
            b, _, h, w = x.shape
            y1, y2, y3 = self.shift(x, b, h, w)
            return self.act(self.shift_bn(y1 + y2 + y3))
        else:
            # 使用 fallback 標準卷積（直接使用 big_k x big_k conv）
            return self.act(self.fallback_bn(self.fallback_conv(x)))

