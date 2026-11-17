import sys
import torch
import torch_npu
import torch_npu.contrib.transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)
sys.path.insert(0, '/work')
from maxpool_bwd_patch import (patch_class, patch_adamw, patch_device_prop,
                               patch_npu_device_index, patch_nan_to_num)
patch_class(); patch_adamw(); patch_device_prop(); patch_npu_device_index(); patch_nan_to_num()
from ultralytics import YOLO
STUDENT = '/work/prune_out/yolo26s-mid-coco.pt'
model = YOLO(STUDENT)
results = model.train(
    data='/work/datasets/mot17_yolo/mot17.yaml', epochs=60, imgsz=640, batch=8, device=0, workers=0,
    project='/work/distill_out', name='mid_thinned_nodistill', exist_ok=True,
    cos_lr=True, plots=False, val=True, amp=False, channels_last=False,
    lr0=0.0005, lrf=0.05, optimizer='auto', patience=60, cache=False,
)
m = results.results_dict
print('=== mid_thinned_nodistill DONE ===')
print('mAP50:', m.get('metrics/mAP50(B)'))
