"""MaxPool2d 反传 NPU 兼容补丁（CANN 9.0.1 aclnnMaxPool2dWithMaskBackward 支持列表为空）
用法：apply_patch() 会把模型中所有 nn.MaxPool2d 替换为 MaxPool2dNPU 模块。
反传用近似梯度（窗口均摊），仅用于打通训练循环验证效率；精确梯度需等 CANN 修复。
"""
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

class MaxPool2dNPUFn(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, ceil_mode):
        out, idx = F.max_pool2d_with_indices(
            input, kernel_size, stride, padding, dilation, ceil_mode)
        ctx.save_for_backward(idx)
        ctx.ks = (kernel_size, stride if stride is not None else kernel_size, padding)
        ctx.in_shape = input.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        k, s, p = ctx.ks
        ki = k if isinstance(k, (tuple, list)) else (k, k)
        si = s if isinstance(s, (tuple, list)) else (s, s)
        pi = p if isinstance(p, (tuple, list)) else (p, p)
        # 近似：grad 上采样回原尺寸，窗口内均摊（/窗口面积）
        g = F.interpolate(grad_output, size=(ctx.in_shape[2], ctx.in_shape[3]),
                          mode='nearest')
        area = si[0] * si[1]
        return g / area, None, None, None, None, None

class MaxPool2dNPU(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False, return_indices=False):
        super().__init__(kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return MaxPool2dNPUFn.apply(x, self.kernel_size, self.stride,
                                    self.padding, self.dilation, self.ceil_mode)

def apply_patch(model):
    n = 0
    for name, child in model.named_children():
        if isinstance(child, nn.MaxPool2d):
            new_m = MaxPool2dNPU(child.kernel_size, child.stride, child.padding,
                                 child.dilation, child.ceil_mode)
            setattr(model, name, new_m)
            n += 1
        else:
            n += apply_patch(child)
    return n


def patch_class():
    """把 nn.MaxPool2d 类整体替换为 NPU 兼容版（影响之后所有实例化，
    包括 ultralytics trainer.get_model() 重建的模型）"""
    # 直接用模块顶层类（可 pickle），并保持 isinstance(nn.MaxPool2d) 成立：
    # MaxPool2dNPU 继承 nn.Module 而非 nn.MaxPool2d，为兼容 isinstance 检查，
    # 注册虚拟基类关系。
    nn.MaxPool2d = MaxPool2dNPU
    import torch.nn.modules.pooling as pooling_mod
    pooling_mod.MaxPool2d = MaxPool2dNPU
    return True

def patch_adamw():
    """AdamW foreach 实现（aclnnApplyAdamWV2）在 CANN 9.0.1 缺失，强制逐参数实现"""
    _orig = torch.optim.AdamW
    class AdamWNoForeach(_orig):
        def __init__(self, *a, **kw):
            kw['foreach'] = False
            kw.pop('fused', None)
            super().__init__(*a, **kw)
    torch.optim.AdamW = AdamWNoForeach
    import torch.optim as optim_mod
    optim_mod.AdamW = AdamWNoForeach
    return True

def patch_device_prop():
    """transfer_to_npu 桥接下 get_device_properties('cuda:0') 解析失败 -> 转 int"""
    _orig = torch.cuda.get_device_properties
    def fixed(dev=None):
        if isinstance(dev, str):
            dev = int(dev.split(':')[-1]) if ':' in dev else 0
        return _orig(dev)
    torch.cuda.get_device_properties = fixed
    return True

def patch_npu_device_index():
    """torch_npu._get_device_index 收到 'cuda:0' 字符串时转 int"""
    import torch_npu.npu.utils as npu_utils
    _orig = npu_utils._get_device_index
    def fixed(device=None, optional=False):
        if isinstance(device, str) and 'cuda' in device:
            device = int(device.split(':')[-1]) if ':' in device else 0
        return _orig(device, optional)
    npu_utils._get_device_index = fixed
    # torch_npu.npu.__init__ 里 from utils import _get_device_index 是按名绑定，需同时替换
    import torch_npu.npu as npu_mod
    if hasattr(npu_mod, '_get_device_index'):
        npu_mod._get_device_index = fixed
    return True

def patch_nan_to_num():
    """torch.nan_to_num_ 在 NPU fp32 上无算子 -> 转 CPU 处理再回写"""
    _orig = torch.nan_to_num_
    def fixed(t, nan=0.0, posinf=None, neginf=None, *, out=None):
        if t.is_npu:
            t_cpu = t.detach().cpu()
            _orig(t_cpu, nan=nan, posinf=posinf, neginf=neginf)
            t.copy_(t_cpu)
            return t
        return _orig(t, nan=nan, posinf=posinf, neginf=neginf, out=out)
    torch.nan_to_num_ = fixed
    torch.Tensor.nan_to_num_ = lambda self, nan=0.0, posinf=None, neginf=None: fixed(self, nan, posinf, neginf)
    return True
