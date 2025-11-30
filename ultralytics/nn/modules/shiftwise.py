"""ShiftWise convolution wrappers for YOLO modules - Pure PyTorch Implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import math


class ShiftWiseConv(nn.Module):
    """ShiftWise convolution module - Pure PyTorch implementation.
    
    This module implements large receptive field convolution using small 3x3 kernels
    with spatial shift patterns, as proposed in the ShiftWise paper.
    
    This is a pure PyTorch implementation that does not require CUDA extensions,
    making it more compatible across different environments.
    
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

        # Save parameters
        self._big_k = big_k
        self._small_k = small_k
        self._c2 = c2
        self._c1 = c1

        # Calculate nk: number of small kernels needed to simulate big_k
        nk = math.ceil(big_k / small_k)  # For big_k=13, small_k=3: nk=5
        self.nk = nk
        c_in_expanded = c2 * nk  # Expanded input channels
        
        # Channel expansion: c1 -> c2 * nk
        self.channel_expand = nn.Conv2d(c1, c_in_expanded, 1, 1, 0, bias=False)
        
        # Generate shift patterns to simulate large kernel receptive field
        # For big_k=13, we need to cover positions from -6 to +6
        # We'll use nk=5 shifts to approximate this
        self.shift_patterns = self._generate_shift_patterns(big_k, small_k, nk)
        
        # Multiple paths: each path uses shifted 3x3 depthwise convs
        # We use 3 paths (matching AddShift_mp_module's output: y1, y2, y3)
        self.num_paths = 3
        self.path_convs = nn.ModuleList()
        for path_idx in range(self.num_paths):
            # Each path: nk depthwise 3x3 convs (one for each shift pattern)
            # All paths share the same shift patterns but have independent conv weights
            path_conv = nn.ModuleList([
                nn.Conv2d(
                    c2, c2, kernel_size=small_k, stride=1, 
                    padding=small_k // 2, groups=c2, bias=False
                ) for _ in range(nk)
            ])
            self.path_convs.append(path_conv)
        
        # Channel mixing: c2 * nk -> c2 (after merging paths)
        self.channel_mix = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        
        # Fallback conv (kept for compatibility, but won't be used in pure PyTorch mode)
        padding = big_k // 2
        self.fallback_conv = nn.Conv2d(c1, c2, big_k, s, padding, bias=False)
        self.fallback_bn = nn.BatchNorm2d(c2)
        
        # Mark as pure PyTorch implementation
        self._path_used = 'pytorch'
        self.use_pytorch = True
        
        # 為每個模組分配 ID（用於日誌識別）
        if not hasattr(ShiftWiseConv, '_instance_count'):
            ShiftWiseConv._instance_count = 0
        ShiftWiseConv._instance_count += 1
        self._module_id = ShiftWiseConv._instance_count

    def _generate_shift_patterns(self, big_k: int, small_k: int, nk: int):
        """Generate shift patterns to simulate large kernel receptive field.
        
        This mimics the shift pattern generation in AddShift_mp_module.shift().
        For big_k=13, small_k=3, nk=5:
        - We need to cover positions to simulate a 13x13 kernel
        - Each of the nk kernels covers a different spatial region
        """
        mid = big_k // 2  # Center position (6 for big_k=13)
        padding = small_k - 1  # = 2 for small_k=3
        
        # Calculate real_pad for each kernel (matching AddShift_mp_module logic)
        # real_pad[i] = mid - i * small_k - padding
        shifts = []
        for i in range(nk):
            extra_pad = mid - i * small_k - padding
            # extra_pad represents the padding offset
            # For shift operation, we use the negative of extra_pad
            # This distributes kernels across the receptive field
            shift_h = -extra_pad
            shift_w = -extra_pad
            shifts.append((shift_h, shift_w))
        
        return shifts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using pure PyTorch implementation.
        
        Process:
        1. Expand channels: c1 -> c2 * nk
        2. For each path:
           - Apply nk shifts with 3x3 depthwise convs
           - Sum the results
        3. Merge paths: sum all path outputs
        4. Channel mixing: c2 -> c2
        
        This implementation uses 3x3 depthwise convolutions with spatial shifts
        to simulate the effect of a large kernel (big_k x big_k).
        """
        if self.stride != 1:
            # Stride > 1 not supported, use fallback
            self._path_used = 'fallback'
            if not hasattr(self, '_fallback_warned'):
                print(f"⚠️  ShiftWiseConv: stride={self.stride} > 1, using fallback (13x13 conv)")
                self._fallback_warned = True
            return self.act(self.fallback_bn(self.fallback_conv(x)))
        
        b, c, h, w = x.shape
        
        # Step 1: Channel expansion
        x_expanded = self.channel_expand(x)  # (b, c1, h, w) -> (b, c2*nk, h, w)
        
        # Reshape to separate channels: (b, c2*nk, h, w) -> (b, c2, nk, h, w)
        x_reshaped = x_expanded.view(b, self._c2, self.nk, h, w)
        
        # Step 2: Process each path (3 paths, each with nk=5 shifts + 3x3 convs)
        path_outputs = []
        for path_idx in range(self.num_paths):
            path_conv = self.path_convs[path_idx]
            path_results = []
            
            # For each shift pattern, apply shift + 3x3 depthwise conv
            for shift_idx, (shift_h, shift_w) in enumerate(self.shift_patterns):
                # Get the corresponding channel group
                x_group = x_reshaped[:, :, shift_idx, :, :]  # (b, c2, h, w)
                
                # Apply spatial shift (this is the key to simulating large kernel)
                if shift_h != 0 or shift_w != 0:
                    x_shifted = torch.roll(x_group, shifts=(shift_h, shift_w), dims=(2, 3))
                else:
                    x_shifted = x_group
                
                # Apply 3x3 depthwise convolution (groups=c2 means depthwise)
                # This is the core: using 3x3 small kernels instead of 13x13 large kernel
                x_conv = path_conv[shift_idx](x_shifted)  # (b, c2, h, w)
                path_results.append(x_conv)
            
            # Sum all shifted conv results for this path
            path_sum = sum(path_results)  # (b, c2, h, w)
            path_outputs.append(path_sum)
        
        # Step 3: Merge all paths (sum)
        merged = sum(path_outputs)  # (b, c2, h, w)
        
        # Step 4: Channel mixing and normalization
        output = self.channel_mix(merged)  # (b, c2, h, w)
        output = self.bn(output)
        
        # Mark path used (for verification)
        self._path_used = 'pytorch'
        
        # Track forward count (for verification)
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
        self._forward_count += 1
        
        # 訓練時輸出驗證資訊（只在第一次或每 N 次輸出）
        if self.training:
            if not hasattr(self, '_training_logged') or self._forward_count % 100 == 0:
                if not hasattr(self, '_training_logged'):
                    # 第一次訓練時輸出
                    module_id = getattr(self, '_module_id', 'unknown')
                    print(f"✅ ShiftWiseConv[{module_id}] 訓練中: 使用 3×3 kernels 模擬 {self._big_k}×{self._big_k}")
                    print(f"   配置: {self.num_paths} paths × {self.nk} shifts × 3×3 depthwise convs")
                    print(f"   輸入: {x.shape} -> 輸出: {output.shape}")
                    self._training_logged = True
        
        return self.act(output)
