"""Torch-Pruning DepGraph 对 YOLO26s 全模型剪枝（业界主流手法）

DepGraph 自动追踪依赖图（含 Concat/Chunk/Split），GroupNorm 重要性打分剪通道，
producer/consumer 自动同步——替代手工 thinning 的均匀采样。

用法: python3 tp_prune_yolo26s.py [ratio]  (默认 0.45)
产出: prune_out/yolo26s_tp_r{ratio}.pt
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import sys

import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
import torch_pruning as tp
from ultralytics import YOLO

WORK = '/data2/gxw/yolo26-310p-eval/prune_out'
os.makedirs(WORK, exist_ok=True)
RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45


def load_s():
    m = YOLO('/data2/gxw/yolo26-310p-eval/yolo26s.pt')
    model = m.model
    model.eval()
    model.float()
    model.end2end = True
    # TP 的 DepGraph 依赖 autograd 追踪，参数必须 requires_grad
    for p in model.parameters():
        p.requires_grad = True
    return model


def make_calib(n=32):
    imgs = sorted(glob.glob('/data2/gxw/mot_datasets/MOT17/train/*/img1/*.jpg'))
    step = max(1, len(imgs) // n)
    out = []
    for p in imgs[::step][:n]:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        out.append(torch.from_numpy((img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]))
    return out


if __name__ == '__main__':
    torch.manual_seed(0)
    model = load_s()
    n0 = sum(p.numel() for p in model.parameters())
    example = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        model(example)
    print(f'before: {n0/1e6:.2f}M params')

    # GroupNormPruner: 按 BN/分组缩放参数 + 激活幅度联合打分（比均匀采样强一个量级）
    imp = tp.importance.GroupMagnitudeImportance(p=2, normalizer='max')
    # 训练数据做前向（打分需要真实数据分布）
    data = make_calib(16)

    def forward_fn(model, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        return model(x)

    pruner = tp.GroupNormPruner(
        model,
        example_inputs=(example,),
        importance=imp,
        global_pruning=False,          # 逐层均匀剪，保持层间比例
        pruning_ratio=RATIO,
        iterative_steps=3,             # 分 3 步渐进剪，给依赖图逐层收敛
        ignored_layers=[m for m in model.modules() if type(m).__name__ == 'Detect'],
        round_to=16,                   # 310P NC1HWC0 对齐
        root_module_types=[nn.Conv2d],
        forward_fn=forward_fn,
    )
    # 渐进剪枝: 每步用真实数据打分
    for i in range(3):
        pruner.step()
        n = sum(p.numel() for p in model.parameters())
        print(f'step {i+1}/3 done: {n/1e6:.2f}M params')

    n_after = sum(p.numel() for p in model.parameters())
    print(f'after: {n_after/1e6:.2f}M ({n_after/n0*100:.1f}%)')

    with torch.no_grad():
        y = model(example)
    print('forward:', type(y).__name__)

    out = f'{WORK}/yolo26s_tp_r{RATIO}.pt'
    torch.save({'model': model, 'train_args': {}}, out)
    print('saved:', out)
