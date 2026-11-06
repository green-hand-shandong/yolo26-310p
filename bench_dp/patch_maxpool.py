import torch
import torch.nn.functional as F
_orig = F.max_pool2d
def patched(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    r = _orig(input.float(), kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    return r
# 已知问题在 backward 的 aclnnMaxPool2dWithMaskBackward 不支持 fp32 —— 
# 实际上 torch_npu 对 fp16 支持好。把 SPPF 模块转 half 不现实。
# 替代：把 maxpool 换成 avgpool 不可行（改结构）。
# 官方推荐：torch.npu.config 的 aclnn 走老接口。设置：
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
