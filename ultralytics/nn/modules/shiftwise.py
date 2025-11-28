"""ShiftWise convolution wrappers for YOLO modules."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ops.ops_py.add_shift import AddShift_mp_module

    HAS_SHIFTWISE = True
except Exception:  # pragma: no cover - optional dependency
    AddShift_mp_module = None
    HAS_SHIFTWISE = False


class ShiftWiseConv(nn.Module):
    """Conv module with ShiftWise CUDA support and full fallback."""

    def __init__(self, c1: int, c2: int, k: int = 5, s: int = 1, act: bool | nn.Module = True):
        super().__init__()
        self.stride = s
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        padding = k // 2
        self.fallback_conv = nn.Conv2d(c1, c2, k, s, padding, bias=False)
        self.fallback_bn = nn.BatchNorm2d(c2)

        self.use_shiftwise = HAS_SHIFTWISE
        if self.use_shiftwise:
            big_k = max(k, 3)
            small_k = 3 if big_k >= 3 else 1
            self.shift = AddShift_mp_module(big_k, small_k, c2, c1, group_in=1)
            self.shift_bn = nn.BatchNorm2d(c2)
        else:
            self.shift = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run shiftwise path when CUDA is available, otherwise fallback to standard conv."""
        if not self.use_shiftwise or not x.is_cuda or self.stride != 1:
            return self.act(self.fallback_bn(self.fallback_conv(x)))

        b, _, h, w = x.shape
        y1, y2, y3 = self.shift(x, b, h, w)
        return self.act(self.shift_bn(y1 + y2 + y3))

