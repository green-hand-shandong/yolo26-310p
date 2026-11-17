import sys
MODE = 'distill'
import torch
import torch_npu
import torch_npu.contrib.transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)
sys.path.insert(0, '/work')
from maxpool_bwd_patch import (patch_class, patch_adamw, patch_device_prop,
                               patch_npu_device_index, patch_nan_to_num)
patch_class(); patch_adamw(); patch_device_prop(); patch_npu_device_index(); patch_nan_to_num()
from ultralytics import YOLO
DATA = '/work/datasets/mot17_yolo/mot17.yaml'
TEACHER = '/work/yolo26s.pt'
STUDENT = '/work/prune_out/yolo26s-mid-coco.pt'
model = YOLO(STUDENT)
results = model.train(
    data=DATA, epochs=60, imgsz=640, batch=8, device=0, workers=0,
    project='/work/distill_out', name='mid_thinned_v2', exist_ok=True,
    cos_lr=True, plots=False, val=True, amp=False, channels_last=False,
    lr0=0.0005, lrf=0.05, optimizer='auto', patience=60, cache=False,
    distill_model=TEACHER, dis=6.0,
)
m = results.results_dict
print('=== mid_thinned_v2 DONE ===')
print('mAP50:', m.get('metrics/mAP50(B)'))
print('mAP50-95:', m.get('metrics/mAP50-95(B)'))
