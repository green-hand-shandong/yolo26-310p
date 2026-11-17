"""310P NPU 蒸馏验证：yolo26s (teacher) → yolo26n (student), MOT17

小规模验证设计（两个实验对照）：
  A. baseline:  yolo26n 直接微调（无蒸馏）
  B. distill:   yolo26n + distill_model=yolo26s.pt（score-weighted L2 特征蒸馏）

复用 maxpool_bwd_patch.py 的 NPU 兼容补丁集（torch_npu 2.9 + CANN 9.0.1 容器内运行）。
用法（容器内）:
  python3 /work/distill_mot17.py baseline
  python3 /work/distill_mot17.py distill
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
STUDENT = '/work/yolo26n.pt'
WORK = '/work/distill_out'

common = dict(
    data=DATA,
    epochs=15,
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
    lr0=0.001,
    lrf=0.01,
    optimizer='auto',
    patience=15,
    cache=False,
)

if MODE == 'baseline':
    name = 'baseline_ft'
    model = YOLO(STUDENT)
    kwargs = {}
else:
    name = 'distill_s2n'
    model = YOLO(STUDENT)
    kwargs = dict(distill_model=TEACHER, dis=6.0)

results = model.train(name=name, exist_ok=True, verbose=True, **common, **kwargs)
m = results.results_dict
print(f'\n=== {MODE} DONE ===')
print('mAP50:', m.get('metrics/mAP50(B)'))
print('mAP50-95:', m.get('metrics/mAP50-95(B)'))
