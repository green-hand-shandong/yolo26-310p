"""探测 ultralytics YOLO26s 中可被 AMCT CNN 剪枝规则识别的 conv 链

AMCT CNN domain 的 target 定义: producer Conv2d → [可选 BatchNorm2d] → consumer Conv2d/Linear，
其中 consumer.in_channels == producer.out_channels、双方 groups==1。
ultralytics 的结构里:
  - Conv 包装模块内部: conv(Conv2d) → bn(BatchNorm2d) → act(SiLU)   [conv 后紧邻 bn ✓]
  - 模块间连接由 forward 驱动，不在 parent 的直接 children 序列里
本脚本统计两类可剪目标:
  A. 模块内链: Conv.conv → Conv.bn（bn→act→下一个模块，consumer 在模块外）
  B. 序列内链: nn.Sequential 内相邻的 conv/bn
另外统计 Concat/Split/C2f 内部 conv 链。
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import sys
sys.path.insert(0, '/data2/gxw/amct')
import torch
from ultralytics import YOLO

m = YOLO('/data2/gxw/yolo26-310p-eval/yolo26s.pt')
model = m.model
model.eval()
model.end2end = True

# 统计所有 Conv2d 及其所在的包装模块
stats = {'total_conv2d': 0, 'in_ul_conv': 0, 'standalone': 0, 'concat_inputs': 0, 'c2f_bottleneck': 0}
conv_pairs = []  # (producer_path, out_c, bn_path, consumer_candidate)
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Conv2d):
        stats['total_conv2d'] += 1
        parent = name.rsplit('.', 1)[0] if '.' in name else ''
        leaf = name.rsplit('.', 1)[-1] if '.' in name else name
        if leaf in ('conv',):
            stats['in_ul_conv'] += 1
        else:
            stats['standalone'] += 1

# 检查 Ultralytics Conv 模块定义（bn 直接跟 conv）
from ultralytics.nn.modules.conv import Conv
c = Conv(8, 16)
print('ultralytics Conv children:', [(k, type(v).__name__) for k, v in c.named_children()])
# Detect head / C2f 结构
det = model.model[-1]
print('head type:', type(det).__name__, '| children:', [type(v).__name__ for k, v in list(det.named_children())[:6]])
c2f = model.model[2]
print('block[2] type:', type(c2f).__name__, '| children:', [type(v).__name__ for k, v in c2f.named_children()][:8])
print('conv2d total:', stats['total_conv2d'], '| inside ultralytics Conv:', stats['in_ul_conv'])

# 关键问题: 模块间 consumer 不在 parent children 里。
# ultralytics Conv 的内部链: conv→bn→act。对 AMCT 的 _find_consumer:
#   conv(idx0) → bn(idx1) ✓ → act=SiLU passthrough → items 尽头 → consumer=None → root linear → None
# 除非... 我们把 wrapper 换成扁平 Sequential！测试:
flat = torch.nn.Sequential(c.conv, c.bn, c.act)
print('flat seq children:', [(k, type(v).__name__) for k, v in flat.named_children()])
