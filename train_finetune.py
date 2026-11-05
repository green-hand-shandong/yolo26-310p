"""310P NPU (torch_npu) YOLO26 微调验证（合成数据集）

NPU 兼容问题（逐项已解）：
1. ultralytics 只认 CUDA API -> torch_npu.contrib.transfer_to_npu 桥接
2. channels_last 默认开 -> channels_last=False
3. ArgMaxWithValue 动态编译失败 -> torch.npu.set_compile_mode(jit_compile=False)
4. MaxPool2d 反传：CANN 9.0.1 aclnnMaxPool2dWithMaskBackward 支持列表为空
   -> nn.MaxPool2d 模块级替换为自定义 autograd（近似梯度）
"""
import torch, torch_npu
import torch_npu.contrib.transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)

import os
os.makedirs('/workspace/datasets/coco128', exist_ok=True)
with open('/workspace/datasets/coco128/coco128.yaml', 'w') as f:
    f.write("""path: /workspace/datasets/coco128
train: images/train2017
val: images/train2017
names:
  0: box
""")

import sys
sys.path.insert(0, '/work')
from maxpool_bwd_patch import patch_class, patch_adamw
patch_class()  # 类级替换：trainer 内部重建的模型也会用 NPU 兼容版
patch_adamw()  # AdamW foreach=False（aclnnApplyAdamWV2 缺失）
from maxpool_bwd_patch import patch_device_prop, patch_npu_device_index, patch_nan_to_num
patch_device_prop()
patch_npu_device_index()  # torch_npu 侧 'cuda:0' 解析修复
patch_nan_to_num()  # nan_to_num_ NPU 无算子

from ultralytics import YOLO
model = YOLO('/work/yolo26n.pt')

results = model.train(
    data='/workspace/datasets/coco128/coco128.yaml',
    epochs=10, imgsz=640, batch=16, device=0, workers=0,
    project='/work/train_out', name='npu_finetune', exist_ok=True,
    verbose=True, lr0=0.001, lrf=0.01, optimizer='auto', cos_lr=True,
    plots=False, val=True, amp=False, channels_last=False,
)
print("\n=== TRAIN DONE ===")
m = results.results_dict
print("mAP50-95:", m.get('metrics/mAP50-95(B)'))
print("mAP50:", m.get('metrics/mAP50(B)'))
