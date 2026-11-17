"""YOLO26 剪枝 domain 适配器：把 ultralytics 的模块图映射成 AMCT CNNChannelTarget

原理：
- ultralytics DetectionModel 按 model.model 列表顺序执行，每个模块带 .f (from 索引) 记录输入来源
- Conv 包装模块内部固定为 conv(Conv2d) → bn(BatchNorm2d) → act(SiLU)
- AMCT CNN 剪枝 target = producer Conv2d → [bn] → consumer Conv2d/Linear（通道数匹配）

target 构建（producer 的输出通道被剪 → 所有一路消费方同步收窄）：
  对每个 ultralytics Conv 模块 i（from=-1，单输入）：
    producer = model[i].conv
    consumer = model[j].conv，其中 j 是"直接消费 i 输出"的下一个模块
  约束：
    - consumer.in_channels == producer.out_channels（1:1 通道直连，无 concat）
    - j 不是 Concat（多输入汇合点，剪 producer 必须同步剪 consumer 的对应输入段，复杂，保守跳过）
    - j 不是 Detect head（head 通道与 e2e 解码耦合，跳过）
    - producer.out_channels > min_channels

保留 AMCT 原生 apply_keep_indices（按 path 替换子模块），只重写 find_targets / channel_count。
"""
from __future__ import annotations

from typing import List

import torch.nn as nn

from amct_pytorch.pruning.config import PruneConfig
from amct_pytorch.pruning.domains.cnn import (
    CNNChannelTarget,
    CNNPruningDomain,
)
from amct_pytorch.pruning.utils import get_submodule


def _unwrap_conv(module: nn.Module):
    """取 ultralytics Conv 包装模块里的 nn.Conv2d；本身是 Conv2d 则原样返回"""
    if isinstance(module, nn.Conv2d):
        return module, None
    conv = getattr(module, 'conv', None)
    if isinstance(conv, nn.Conv2d):
        return conv, getattr(module, 'bn', None)
    return None, None


class YOLO26PruningDomain(CNNPruningDomain):
    """针对 ultralytics DetectionModel 的通道剪枝 domain（单输入 Conv 链）"""

    name = 'cnn'  # 复用 cnn 域的方法绑定（variance_channel / reconstruct）

    def find_targets(self, model: nn.Module, config: PruneConfig) -> List[CNNChannelTarget]:
        seq = getattr(model, 'model', None)
        if seq is None:
            return super().find_targets(model, config)

        # 建反向消费图: producer_idx -> [consumer_idx...]
        # f=-1 表示取上一个模块的输出（线性链）；f=[...] 为显式多输入
        consumers: dict[int, list[int]] = {}
        for j, module in enumerate(seq):
            f = getattr(module, 'f', -1)
            if isinstance(f, int):
                if f == -1:
                    if j >= 1:
                        consumers.setdefault(j - 1, []).append(j)
                elif f >= 0:
                    consumers.setdefault(f, []).append(j)
            elif isinstance(f, (list, tuple)):
                for fi in f:
                    if isinstance(fi, int) and fi >= 0:
                        consumers.setdefault(fi, []).append(j)

        targets: List[CNNChannelTarget] = []
        for i, module in enumerate(seq):
            if type(module).__name__ not in ('Conv', 'ConvDW'):  # 只剪标准 Conv 包装
                continue
            producer, bn = _unwrap_conv(module)
            if producer is None or producer.groups != 1:
                continue
            if producer.out_channels <= config.min_channels:
                continue
            # 单输入、单消费者
            outs = consumers.get(i, [])
            if len(outs) != 1:
                continue
            j = outs[0]
            cmod = seq[j]
            ctype = type(cmod).__name__
            # consumer 解析：
            #   Conv 包装      -> cmod.conv            （直连）
            #   C3k2/C2f 系列  -> cmod.cv1.conv        （块的第一个卷积）
            #   Concat/Detect/SPPF/C2PSA       -> 跳过（汇合/敏感点）
            if ctype == 'Conv':
                consumer, cbn = _unwrap_conv(cmod)
                consumer_path = f'model.{j}.conv'
            elif ctype in ('C3k2', 'C2f', 'C3', 'Bottleneck'):
                inner = getattr(cmod, 'cv1', None)
                consumer, cbn = _unwrap_conv(inner) if inner is not None else (None, None)
                consumer_path = f'model.{j}.cv1.conv'
            else:
                continue
            if consumer is None or consumer.groups != 1:
                continue
            if consumer.in_channels != producer.out_channels:
                continue
            targets.append(CNNChannelTarget(
                producer_path=f'model.{i}.conv',
                bn_path=f'model.{i}.bn' if bn is not None else None,
                consumer_path=consumer_path,
            ))
        return targets

    def channel_count(self, model: nn.Module, target: CNNChannelTarget) -> int:
        producer = get_submodule(model, target.producer_path)
        if not isinstance(producer, nn.Conv2d):
            raise TypeError(f"Target producer '{target.producer_path}' is not Conv2d.")
        return int(producer.out_channels)
