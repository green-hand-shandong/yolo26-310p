"""yolo26s 剪枝 → 中间档模型（AMCT variance_channel + YOLO26 domain 适配器）

用自定义注册表（YOLO26PruningDomain 替换默认 cnn domain）跑固定比率剪枝。
产出: prune_out/yolo26s_pruned.pt（可继续蒸馏的中间档模型）
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import sys
sys.path.insert(0, '/data2/gxw/amct')
sys.path.insert(0, '/data2/gxw/yolo26-310p-eval')

import glob
import torch
import cv2
import numpy as np
from ultralytics import YOLO

import amct_pytorch.pruning as pr
from amct_pytorch.pruning.pruner import AutoPruner
from amct_pytorch.pruning.registry import create_default_registry
from amct_pytorch.pruning.config import PruneConfig
from yolo26_prune_domain import YOLO26PruningDomain

WORK = '/data2/gxw/yolo26-310p-eval/prune_out'
os.makedirs(WORK, exist_ok=True)

RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45


def load_s():
    m = YOLO('/data2/gxw/yolo26-310p-eval/yolo26s.pt')
    model = m.model
    model.eval()
    model.float()
    model.end2end = True   # 不 fuse：保留 BN 供剪枝同步收窄
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
    n_before = sum(p.numel() for p in model.parameters())
    data = make_calib(32)
    with torch.no_grad():
        model(data[0])
    print(f'before: {n_before/1e6:.2f}M params')

    cfg = PruneConfig(**{
        'methods': {'cnn': {'name': 'variance_channel', 'kwargs': {'prune_ratio': RATIO}}},
        'missing_data_policy': 'raise',
    })
    registry = create_default_registry()
    # 用 YOLO26 domain 替换默认 cnn domain（方法绑定不变）
    registry = {(d, m): b for (d, m), b in registry.items()}
    for (d, m_name), binding in list(registry.items()):
        if d == 'cnn':
            registry[(d, m_name)] = type(binding)(YOLO26PruningDomain(), binding.method)

    pruner = AutoPruner(cfg, registry=registry)
    pruned = pruner(model, data=data)
    n_after = sum(p.numel() for p in pruned.parameters())
    print(f'after:  {n_after/1e6:.2f}M params ({n_after/n_before*100:.1f}%)')
    print('--- report ---')
    for entry in pruner.last_report.entries if hasattr(pruner.last_report, 'entries') else []:
        print(entry)

    # 保存为 ultralytics 兼容 ckpt
    ckpt = {
        'model': pruned,
        'train_args': {},
    }
    out = f'{WORK}/yolo26s_pruned_r{RATIO}.pt'
    torch.save(ckpt, out)
    print('saved:', out)

    # 前向 sanity
    with torch.no_grad():
        y = pruned(data[0])
    print('forward after prune:', type(y).__name__)
