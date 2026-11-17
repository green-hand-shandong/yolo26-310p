"""yolo26s-mid 蒸馏训练：teacher=yolo26s → student=yolo26s-mid（随机初始化）

从 yolo26 官方缩放体系生成宽度 0.375 的中间档结构（5.65M，介于 n 2.57M / s 10.0M），
用 yolo26s 做 teacher 在 MOT17 上蒸馏训练。
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'distill'

import torch
import torch_npu
import torch_npu.contrib.transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)

sys.path.insert(0, '/work')
from maxpool_bwd_patch import (patch_class, patch_adamw, patch_device_prop,
                               patch_npu_device_index, patch_nan_to_num)
patch_class()
patch_adamw()
patch_device_prop()
patch_npu_device_index()
patch_nan_to_num()

from ultralytics import YOLO

DATA = '/work/datasets/mot17_yolo/mot17.yaml'
TEACHER = '/work/yolo26s.pt'
STUDENT_CFG = '/work/datasets/mot17_yolo/yolo26s-mid.yaml'
WORK = '/work/distill_out'

common = dict(
    data=DATA,
    epochs=40,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    project=WORK,
    cos_lr=True,
    plots=False,
    val=True,
    amp=False,
    channels_last=False,
    lr0=0.002,
    lrf=0.05,
    optimizer='auto',
    patience=40,
    cache=False,
)

if MODE == 'distill':
    name = 'mid_distill_s2mid'
    model = YOLO(STUDENT_CFG)
    kwargs = dict(distill_model=TEACHER, dis=6.0)
elif MODE == 'baseline':
    name = 'mid_baseline'
    model = YOLO(STUDENT_CFG)
    kwargs = {}
else:
    raise ValueError(MODE)

results = model.train(name=name, exist_ok=True, verbose=True, **common, **kwargs)
m = results.results_dict
print(f'\n=== {MODE} DONE ===')
print('mAP50:', m.get('metrics/mAP50(B)'))
print('mAP50-95:', m.get('metrics/mAP50-95(B)'))
