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
            # AddShift_mp_module 需要：c_in = c_out * nk，其中 nk = ceil(big_k / small_k)
            import math
            nk = math.ceil(big_k / small_k)  # 對於 big_k=13, small_k=3: nk=5
            c_in_expanded = c2 * nk  # 擴展後的輸入通道數
            
            # 初始化：AddShift_mp_module(big_kernel, small_kernel, c_out, c_in, group_in)
            # c_out = c2 (輸出通道)
            # c_in = c_in_expanded = c2 * nk (擴展後的輸入通道)
            self.shift = shift_module(big_k, small_k, c2, c_in_expanded, group_in=1)
            self.shift_bn = nn.BatchNorm2d(c2)
            self.nk = nk  # 保存 nk 以便在 forward 中使用
            
            # 需要一個 1x1 conv 來將輸入從 c1 擴展到 c_in_expanded
            self.channel_expand = nn.Conv2d(c1, c_in_expanded, 1, 1, 0, bias=False)
        else:
            self.shift = None
            self.shift_bn = None
            self.nk = None
            self.channel_expand = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run shiftwise path when CUDA is available, otherwise fallback to standard conv."""
        # 檢查是否可以使用 ShiftWise CUDA 路徑
        if (
            self.use_shiftwise
            and self.shift is not None
            and self.channel_expand is not None
            and x.is_cuda
            and self.stride == 1
        ):
            # 使用 ShiftWise CUDA 路徑（3x3 kernels + shift pattern 實現等效 big_k）
            b, c, h, w = x.shape
            
            # AddShift_mp_module 需要輸入通道數為 c_out * nk
            # 所以我們需要先擴展通道數
            try:
                # 確保張量是連續的
                if not x.is_contiguous():
                    x = x.contiguous()
                
                # 擴展通道數：從 c1 擴展到 c2 * nk
                x_expanded = self.channel_expand(x)  # (b, c1, h, w) -> (b, c2*nk, h, w)
                
                # 計算 extra_pad（與 AddShift_mp_module 內部計算一致）
                # extra_pad = (small_k - 1) - small_k // 2
                small_k = 3  # 固定為 3
                extra_pad = (small_k - 1) - small_k // 2  # = 1
                
                # 計算輸出尺寸
                # AddShift_mp_module 內部：x_hin = hout + 2*extra_pad
                # 所以 hout = h - 2*extra_pad
                hout = h - 2 * extra_pad
                wout = w - 2 * extra_pad
                
                # 確保輸出尺寸是正數
                if hout <= 0 or wout <= 0:
                    raise ValueError(f"Invalid output size: hout={hout}, wout={wout} (input: h={h}, w={w})")
                
                # 確保擴展後的張量是連續的
                if not x_expanded.is_contiguous():
                    x_expanded = x_expanded.contiguous()
                
                # 調用 ShiftWise CUDA kernel
                y1, y2, y3 = self.shift(x_expanded, b, hout, wout)
                
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

