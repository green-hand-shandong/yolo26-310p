"""yolo26s 剪枝可行性诊断（CPU，dry-run，不改原模型）

跑三个剪枝率 (0.3/0.45/0.6) 的 dry-run，看 AMCT CNN 通道剪枝
能剪到 yolo26n 与 yolo26s 之间的哪个档位，以及精度容差表现。
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import sys
sys.path.insert(0, '/data2/gxw/amct')

import glob
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import amct_pytorch as amct
from amct_pytorch.pruning import prune_diagnose, CNN_VARIANCE_PRUNE_CFG

WORK = '/data2/gxw/yolo26-310p-eval/prune_out'
os.makedirs(WORK, exist_ok=True)


def load_s_model():
    m = YOLO('/data2/gxw/yolo26-310p-eval/yolo26s.pt')
    model = m.model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model.end2end = True
    model = model.fuse(imgsz=640)
    return model


def make_calib(n=24):
    imgs = sorted(glob.glob('/data2/gxw/mot_datasets/MOT17/train/*/img1/*.jpg'))
    step = max(1, len(imgs) // n)
    out = []
    for p in imgs[::step][:n]:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out.append(torch.from_numpy(x))
    return out


def count_params_flops(model):
    n_params = sum(p.numel() for p in model.parameters())
    return n_params


if __name__ == '__main__':
    torch.manual_seed(0)
    model = load_s_model()
    n0 = count_params_flops(model)
    print(f'yolo26s (fused e2e): {n0/1e6:.2f}M params')
    data = make_calib(24)
    with torch.no_grad():
        model(data[0])
    print('forward smoke ok')

    for ratio in (0.3, 0.45, 0.6):
        print(f'\n=== diagnose ratio={ratio} ===')
        try:
            rep = prune_diagnose(model, data=data, prune_ratio=ratio, tolerance=0.05)
            print(rep)
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {str(e)[:300]}')
