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
    except ImportError as e:
        # Import 失敗，可能是模組未安裝或未編譯
        if not hasattr(_check_shiftwise_available, '_import_warned'):
            print(f"⚠️  ShiftWise CUDA module import failed: {e}")
            print(f"   請確認 shift-wiseConv 已正確編譯和安裝")
            _check_shiftwise_available._import_warned = True
        return False, None
    except Exception as e:
        # 其他錯誤
        if not hasattr(_check_shiftwise_available, '_import_warned'):
            print(f"⚠️  ShiftWise CUDA module check failed: {type(e).__name__}: {e}")
            _check_shiftwise_available._import_warned = True
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

        # 保存參數（無論 ShiftWise 是否可用，都需要這些參數以便後續重新檢查）
        self._big_k = big_k
        self._small_k = small_k
        self._c2 = c2
        self._c1 = c1

        # 檢查環境變數：如果設置了 SHIFTWISE_DISABLE=1，完全禁用 ShiftWise
        import os
        shiftwise_disabled = os.getenv("SHIFTWISE_DISABLE", "0") == "1"
        
        # 動態檢查 ShiftWise CUDA 模組是否可用（每次初始化時重新檢查）
        self.use_shiftwise, shift_module = _check_shiftwise_available()
        if shiftwise_disabled:
            self.use_shiftwise = False
            if not hasattr(self, '_disable_warned'):
                print("⚠️  ShiftWise 已通過環境變數禁用 (SHIFTWISE_DISABLE=1)")
                self._disable_warned = True
        
        if self.use_shiftwise and shift_module is not None:
            # AddShift_mp_module 需要：c_in = c_out * nk，其中 nk = ceil(big_k / small_k)
            import math
            nk = math.ceil(big_k / small_k)  # 對於 big_k=13, small_k=3: nk=5
            c_in_expanded = c2 * nk  # 擴展後的輸入通道數
            
            # 保存初始化參數，延遲初始化 AddShift_mp_module
            # 因為 AddShift_mp_module.__init__ 會調用 torch.manual_seed，可能觸發 CUDA 操作
            # 如果 CUDA 未正確初始化會導致錯誤
            self._shift_module_class = shift_module
            self._c_in_expanded = c_in_expanded
            self.shift = None  # 延遲初始化
            self.shift_bn = nn.BatchNorm2d(c2)
            self.nk = nk  # 保存 nk 以便在 forward 中使用
            
            # 需要一個 1x1 conv 來將輸入從 c1 擴展到 c_in_expanded
            self.channel_expand = nn.Conv2d(c1, c_in_expanded, 1, 1, 0, bias=False)
        else:
            # ShiftWise 不可用，但保存參數以便後續重新檢查
            self.shift = None
            self.shift_bn = None
            self.nk = None
            self.channel_expand = None
            self._shift_module_class = None
            self._c_in_expanded = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run shiftwise path when CUDA is available, otherwise fallback to standard conv."""
        # 重新檢查 ShiftWise 是否可用（可能在 __init__ 時不可用，但現在可用了）
        if not self.use_shiftwise or self._shift_module_class is None:
            # 如果之前不可用，現在重新檢查
            use_shiftwise_new, shift_module = _check_shiftwise_available()
            if use_shiftwise_new and shift_module is not None:
                # 現在可用，設置參數
                import math
                nk = math.ceil(self._big_k / self._small_k)
                c_in_expanded = self._c2 * nk
                self._shift_module_class = shift_module
                self._c_in_expanded = c_in_expanded
                self.nk = nk
                self.use_shiftwise = True
                if not hasattr(self, 'shift_bn') or self.shift_bn is None:
                    self.shift_bn = nn.BatchNorm2d(self._c2)
                if not hasattr(self, 'channel_expand') or self.channel_expand is None:
                    self.channel_expand = nn.Conv2d(
                        self._c1, c_in_expanded, 1, 1, 0, bias=False
                    )
                if not hasattr(self, '_reinit_warned'):
                    print(f"✅ ShiftWise CUDA 模組現在可用，已啟用 ShiftWise 路徑")
                    self._reinit_warned = True
        
        # 延遲初始化 AddShift_mp_module（在第一次 forward 時，確保 CUDA 已正確初始化）
        if self.use_shiftwise and self.shift is None and self._shift_module_class is not None:
            # 只在 CUDA 輸入時初始化（ShiftWise 需要 CUDA）
            if x.is_cuda:
                try:
                    # 確保 CUDA 已初始化並清除任何之前的錯誤狀態
                    torch.cuda.synchronize()
                    # 檢查 CUDA 是否可用
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is not available")
                    
                    # 現在初始化 AddShift_mp_module
                    # 注意：AddShift_mp_module.__init__ 會調用 torch.manual_seed，可能觸發 CUDA 操作
                    # 使用 CUDA_LAUNCH_BLOCKING 來獲取更詳細的錯誤信息
                    import os
                    old_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "0")
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 臨時啟用以獲取詳細錯誤
                    
                    try:
                        self.shift = self._shift_module_class(
                            self._big_k, self._small_k, self._c2, self._c_in_expanded, group_in=1
                        )
                        # 將 shift 模組移到 CUDA
                        self.shift = self.shift.cuda()
                    finally:
                        # 恢復原來的設置
                        os.environ["CUDA_LAUNCH_BLOCKING"] = old_blocking
                    
                except (RuntimeError, Exception) as e:
                    # 如果初始化失敗，禁用 ShiftWise 並打印詳細錯誤
                    error_msg = str(e)
                    error_type = type(e).__name__
                    
                    # 總是打印第一次失敗的詳細信息（每個模組實例）
                    if not hasattr(self, '_init_warned'):
                        print(f"\n{'='*60}")
                        print(f"⚠️  ShiftWise CUDA module initialization failed")
                        print(f"{'='*60}")
                        print(f"Module: {self.__class__.__name__}")
                        print(f"Error type: {error_type}")
                        print(f"Error message: {error_msg}")
                        print(f"Parameters: big_k={self._big_k}, small_k={self._small_k}, c2={self._c2}, c_in_expanded={self._c_in_expanded}")
                        print(f"CUDA available: {torch.cuda.is_available()}")
                        if torch.cuda.is_available():
                            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                            print(f"CUDA version: {torch.version.cuda}")
                        print(f"PyTorch version: {torch.__version__}")
                        print(f"\n可能的原因:")
                        print(f"1. shift-wiseConv CUDA kernel 編譯問題")
                        print(f"2. CUDA/PyTorch 版本不相容")
                        print(f"3. CUDA 上下文損壞")
                        print(f"4. 記憶體不足")
                        print(f"\n將使用 fallback 標準卷積")
                        print(f"{'='*60}\n")
                        self._init_warned = True
                        self._init_error = error_msg  # 保存錯誤信息以便後續查詢
                    
                    self.use_shiftwise = False
                    self.shift = None
            else:
                # CPU 輸入，不使用 ShiftWise
                if not hasattr(self, '_cpu_warned'):
                    print(f"⚠️  Input is on CPU, ShiftWise requires CUDA. Using fallback.")
                    self._cpu_warned = True
                self.use_shiftwise = False
                self.shift = None
        
        # 檢查是否可以使用 ShiftWise CUDA 路徑
        use_shiftwise_path = (
            self.use_shiftwise
            and self.shift is not None
            and self.channel_expand is not None
            and x.is_cuda
            and self.stride == 1
        )
        
        # 標記：記錄是否使用 ShiftWise 路徑（用於驗證）
        if not hasattr(self, '_path_used'):
            self._path_used = None  # None=未使用, 'shiftwise'=使用 ShiftWise, 'fallback'=使用 fallback
        
        if use_shiftwise_path:
            # 使用 ShiftWise CUDA 路徑（3x3 kernels + shift pattern 實現等效 big_k）
            b, c, h, w = x.shape
            
            # AddShift_mp_module 需要輸入通道數為 c_out * nk
            # 所以我們需要先擴展通道數
            try:
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
                
                # AddShift_mp_module 期望輸入尺寸為 (b, c_in, x_hin, x_win)
                # 其中 x_hin = hout + 2*extra_pad = h, x_win = wout + 2*extra_pad = w
                # 所以輸入的 h, w 應該等於 x_hin, x_win
                # 但我們需要確保輸入尺寸正確
                x_hin_expected = hout + 2 * extra_pad
                x_win_expected = wout + 2 * extra_pad
                
                # 檢查輸入尺寸是否匹配
                if h != x_hin_expected or w != x_win_expected:
                    raise ValueError(
                        f"Input size mismatch: expected (h={x_hin_expected}, w={x_win_expected}), "
                        f"got (h={h}, w={w}), hout={hout}, wout={wout}, extra_pad={extra_pad}"
                    )
                
                # 確保張量是連續的
                if not x.is_contiguous():
                    x = x.contiguous()
                
                # 擴展通道數：從 c1 擴展到 c2 * nk
                x_expanded = self.channel_expand(x)  # (b, c1, h, w) -> (b, c2*nk, h, w)
                
                # 確保擴展後的張量是連續的
                if not x_expanded.is_contiguous():
                    x_expanded = x_expanded.contiguous()
                
                # 調用 ShiftWise CUDA kernel
                # AddShift_mp_module.forward(x, b, hout, wout)
                # 內部會計算 x_hin = hout + 2*extra_pad，應該等於我們的 h
                try:
                    # 驗證輸入參數
                    if b <= 0 or hout <= 0 or wout <= 0:
                        raise ValueError(f"Invalid parameters: b={b}, hout={hout}, wout={wout}")
                    if x_expanded.shape[0] != b:
                        raise ValueError(f"Batch size mismatch: x.shape[0]={x_expanded.shape[0]}, b={b}")
                    if x_expanded.shape[1] != self.shift.c_in:
                        raise ValueError(
                            f"Channel mismatch: x.shape[1]={x_expanded.shape[1]}, "
                            f"expected c_in={self.shift.c_in}"
                        )
                    
                    y1, y2, y3 = self.shift(x_expanded, b, hout, wout)
                except RuntimeError as e:
                    # 如果 CUDA kernel 出錯，提供更詳細的錯誤信息
                    error_msg = str(e)
                    if "CUDA" in error_msg or "cuda" in error_msg.lower() or "illegal" in error_msg.lower():
                        print(f"⚠️  ShiftWise CUDA kernel error:")
                        print(f"   Input shape: {x_expanded.shape}")
                        print(f"   Expected input: (b={b}, c_in={self.shift.c_in}, h={x_hin_expected}, w={x_win_expected})")
                        print(f"   Output size: hout={hout}, wout={wout}")
                        print(f"   Error: {error_msg}")
                        raise
                    else:
                        raise
                
                # 檢查輸出形狀
                expected_shape = (b, self.shift.c_out, hout, wout)
                if y1.shape != expected_shape or y2.shape != expected_shape or y3.shape != expected_shape:
                    raise ValueError(
                        f"Output shape mismatch: expected {expected_shape}, "
                        f"got y1={y1.shape}, y2={y2.shape}, y3={y3.shape}"
                    )
                
                # 同步 CUDA 操作以檢查是否有錯誤（在相加之前）
                torch.cuda.synchronize()
                
                # 檢查輸出張量是否有效
                try:
                    # 嘗試訪問張量的 shape 來檢查是否有效
                    _ = y1.shape
                    _ = y2.shape
                    _ = y3.shape
                except RuntimeError as e:
                    # 如果張量無效，CUDA 上下文可能已損壞
                    error_msg = str(e)
                    if "CUDA" in error_msg or "cuda" in error_msg.lower():
                        # 嘗試重置 CUDA 上下文
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        except:
                            pass
                        raise RuntimeError(f"CUDA context corrupted after ShiftWise kernel: {error_msg}")
                    raise
                
                # 確保輸出張量是連續的
                result = y1 + y2 + y3
                if not result.is_contiguous():
                    result = result.contiguous()
                
                # ShiftWise 輸出尺寸會比輸入小 2*extra_pad
                # 但 YOLO 架構期望輸出尺寸等於輸入尺寸（stride=1, padding=k//2）
                # 所以我們需要進行 padding 來恢復原始尺寸
                if hout != h or wout != w:
                    # 計算需要的 padding
                    pad_h = (h - hout) // 2
                    pad_w = (w - wout) // 2
                    pad_h_remainder = (h - hout) % 2
                    pad_w_remainder = (w - wout) % 2
                    
                    # 進行 padding：pad (left, right, top, bottom)
                    result = torch.nn.functional.pad(
                        result,
                        (pad_w, pad_w + pad_w_remainder, pad_h, pad_h + pad_h_remainder),
                        mode='constant',
                        value=0
                    )
                
                # 同步 CUDA 操作以檢查是否有錯誤
                torch.cuda.synchronize()
                
                # 標記：成功使用 ShiftWise 路徑
                self._path_used = 'shiftwise'
                
                return self.act(self.shift_bn(result))
            except (RuntimeError, ValueError) as e:
                # 如果 CUDA kernel 出錯或尺寸不匹配，fallback 到標準卷積
                error_msg = str(e)
                if "CUDA" in error_msg or "cuda" in error_msg.lower() or "illegal" in error_msg.lower():
                    # 只在第一次出錯時打印警告，避免刷屏
                    if not hasattr(self, '_fallback_warned'):
                        print(f"⚠️  ShiftWise CUDA kernel error, falling back to standard conv: {error_msg}")
                        print(f"   這可能是 CUDA kernel 與當前 PyTorch/CUDA 版本不相容")
                        print(f"   建議：重新編譯 shift-wiseConv 或使用 fallback 模式")
                        self._fallback_warned = True
                    
                    # 嘗試清除 CUDA 錯誤狀態
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except:
                        pass
                    
                    # 永久禁用 ShiftWise 以避免後續錯誤
                    self.use_shiftwise = False
                    self.shift = None
                    
                    # 標記：因為錯誤而使用 fallback
                    self._path_used = 'fallback'
                    
                    # 確保輸入在正確的設備上
                    if not x.is_cuda and torch.cuda.is_available():
                        x = x.cuda()
                    elif x.is_cuda and not torch.cuda.is_available():
                        x = x.cpu()
                    
                    return self.act(self.fallback_bn(self.fallback_conv(x)))
                else:
                    raise
        else:
            # 使用 fallback 標準卷積（直接使用 big_k x big_k conv）
            # 標記：使用 fallback 路徑
            self._path_used = 'fallback'
            
            return self.act(self.fallback_bn(self.fallback_conv(x)))

