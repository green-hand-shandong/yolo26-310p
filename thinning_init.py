"""通道子采样初始化（channel thinning）：yolo26s (COCO) → yolo26s-mid

原理：mid 与 s 共享同一 yaml 拓扑（depth 相同），仅宽度不同（0.375 vs 0.50）。
对每个卷积层，把 s 的权重沿输出通道均匀采样（每 4 取 3）到 mid 的通道数；
输入通道按"该层输入来源层的输出通道选择"同步采样，保证生产者-消费者对齐。
BN 参数随输出通道采样。Detect head 的分支层按各自输入对齐。

产出: yolo26s-mid-coco.pt —— mid 结构 + s 的 COCO 知识（75% 通道）
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
import torch.nn as nn
from ultralytics import YOLO

SRC = '/data2/gxw/yolo26-310p-eval/yolo26s.pt'
MID_CFG = '/data2/gxw/datasets/mot17_yolo/yolo26s-mid.yaml'
OUT = '/data2/gxw/yolo26-310p-eval/prune_out/yolo26s-mid-coco.pt'


def even_keep(n_src, n_dst):
    """均匀采样索引: 从 n_src 个通道里取 n_dst 个（保首尾）"""
    if n_dst >= n_src:
        return list(range(n_src))
    return sorted(set(round(i * (n_src - 1) / (n_dst - 1)) for i in range(n_dst))) if n_dst > 1 else [0]


def main():
    src_ckpt = torch.load(SRC, map_location='cpu', weights_only=False)
    src_model = src_ckpt['model'].float()
    src_model.eval()
    src_model.end2end = True

    mid = YOLO(MID_CFG)
    mid_model = mid.model
    mid_model.eval()
    mid_model.end2end = True

    src_seq = src_model.model
    mid_seq = mid_model.model
    assert len(src_seq) == len(mid_seq), f'拓扑不一致: {len(src_seq)} vs {len(mid_seq)}'

    # 对每个模块索引，计算 s → mid 的输出通道选择
    # out_keep[i]: src 模块 i 的输出通道 → mid 模块 i 的输入通道选择
    n_layers = len(src_seq)
    out_keep = {}
    for i in range(n_layers):
        src_conv = getattr(src_seq[i], 'conv', None)
        mid_conv = getattr(mid_seq[i], 'conv', None)
        if src_conv is None or mid_conv is None:
            continue
        n_src, n_dst = src_conv.out_channels, mid_conv.out_channels
        if n_src == n_dst:
            out_keep[i] = list(range(n_src))
        else:
            out_keep[i] = even_keep(n_src, n_dst)
    # C3k2/C2f/SPPF/C2PSA 的输出 = cv2 输出，供下游 producer→consumer 对齐
    # C3k2/C2f 的 cv1 输出被 chunk(2)：前半直连 cv2、后半进 m 链。
    # 输出 keep 必须两半各自采样同一索引序列，保持 chunk 语义。
    cv1_out_keep = {}
    for i in range(n_layers):
        s_mod, m_mod = src_seq[i], mid_seq[i]
        s_cv2, m_cv2 = getattr(s_mod, 'cv2', None), getattr(m_mod, 'cv2', None)
        if s_cv2 is not None and m_cv2 is not None and type(s_mod).__name__ in ('C3k2', 'C2f'):
            out_keep[i] = even_keep(s_cv2.conv.out_channels, m_cv2.conv.out_channels)
        s_cv1, m_cv1 = getattr(s_mod, 'cv1', None), getattr(m_mod, 'cv1', None)
        if s_cv1 is not None and m_cv1 is not None and type(s_mod).__name__ in ('C3k2', 'C2f'):
            half_s = s_cv1.conv.out_channels // 2
            half_m = m_cv1.conv.out_channels // 2
            half_idx = even_keep(half_s, half_m)
            cv1_out_keep[i] = half_idx + [h + half_s for h in half_idx]

    def src_sd(name):
        return dict(src_model.named_parameters()), dict(src_model.named_buffers())

    src_params = dict(src_model.named_parameters())
    src_buffers = dict(src_model.named_buffers())
    mid_params = dict(mid_model.named_parameters())
    mid_buffers = dict(mid_model.named_buffers())

    transferred = 0
    skipped = []

    # 逐模块迁移
    for i in range(n_layers):
        s_mod, m_mod = src_seq[i], mid_seq[i]
        keep_out = out_keep.get(i)

        # Conv 包装: conv / bn
        s_conv, m_conv = getattr(s_mod, 'conv', None), getattr(m_mod, 'conv', None)
        if isinstance(s_conv, nn.Conv2d) and isinstance(m_conv, nn.Conv2d):
            # 输入通道选择: 该模块的输入来自哪个 producer
            f = getattr(s_mod, 'f', -1)
            if isinstance(f, int):
                src_prod = f if f >= 0 else i - 1
            else:
                src_prod = None
            if src_prod is not None and src_prod in out_keep:
                keep_in = out_keep[src_prod]
            else:
                keep_in = list(range(m_conv.in_channels)) if m_conv.in_channels == s_conv.in_channels else None

            w = s_conv.weight.data
            if keep_in is not None and len(keep_in) == m_conv.in_channels:
                if len(keep_out) == m_conv.out_channels:
                    m_conv.weight.data.copy_(w[keep_out][:, keep_in])
                    transferred += 1
                else:
                    skipped.append(f'{i}.conv.weight shape {tuple(m_conv.weight.shape)}')
            else:
                skipped.append(f'{i}.conv (in mismatch {s_conv.in_channels} vs {m_conv.in_channels})')
            if s_conv.bias is not None and m_conv.bias is not None and len(keep_out) == m_conv.out_channels:
                m_conv.bias.data.copy_(s_conv.bias.data[keep_out])

            s_bn, m_bn = getattr(s_mod, 'bn', None), getattr(m_mod, 'bn', None)
            if isinstance(s_bn, nn.BatchNorm2d) and isinstance(m_bn, nn.BatchNorm2d) and len(keep_out) == m_bn.num_features:
                for attr in ('weight', 'bias', 'running_mean', 'running_var'):
                    t = getattr(s_bn, attr).data
                    getattr(m_bn, attr).data.copy_(t[keep_out])
                m_bn.num_batches_tracked.data.copy_(s_bn.num_batches_tracked.data)

    # C3k2 内部（cv1/cv2/m 链）: 结构相同仅宽度不同，同样按通道采样
    for i in range(n_layers):
        s_mod, m_mod = src_seq[i], mid_seq[i]
        if type(s_mod).__name__ not in ('C3k2', 'C2f', 'SPPF', 'C2PSA'):
            continue
        # cv1: 输入来自 producer（out_keep 已有），输出 = cv1.out
        s_cv1, m_cv1 = getattr(s_mod, 'cv1', None), getattr(m_mod, 'cv1', None)
        if s_cv1 is None or m_cv1 is None:
            continue
        f = getattr(s_mod, 'f', -1)
        src_prod = f if (isinstance(f, int) and f >= 0) else (i - 1 if isinstance(f, int) else None)
        keep_in = out_keep.get(src_prod) if src_prod is not None else None
        keep_cv1_out = cv1_out_keep.get(i, even_keep(s_cv1.conv.out_channels, m_cv1.conv.out_channels))
        if keep_in is not None and len(keep_in) == m_cv1.conv.in_channels:
            m_cv1.conv.weight.data.copy_(s_cv1.conv.weight.data[keep_cv1_out][:, keep_in])
            transferred += 1
            s_bn, m_bn = getattr(s_cv1, 'bn', None), getattr(m_cv1, 'bn', None)
            if isinstance(s_bn, nn.BatchNorm2d) and isinstance(m_bn, nn.BatchNorm2d):
                for attr in ('weight', 'bias', 'running_mean', 'running_var'):
                    getattr(m_bn, attr).data.copy_(getattr(s_bn, attr).data[keep_cv1_out])
        # m 链 (Bottleneck): cv1/cv2 链式，输入通道 = cv1_out/2
        if isinstance(getattr(s_mod, 'm', None), nn.ModuleList) and isinstance(getattr(m_mod, 'm', None), nn.ModuleList):
            mchain_prev = None
            for bi, (s_b, m_b) in enumerate(zip(s_mod.m, m_mod.m)):
                sb1, mb1 = getattr(s_b, 'cv1', None), getattr(m_b, 'cv1', None)
                sb2, mb2 = getattr(s_b, 'cv2', None), getattr(m_b, 'cv2', None)
                if None in (sb1, mb1, sb2, mb2):
                    continue
                k1o = even_keep(sb1.conv.out_channels, mb1.conv.out_channels)
                k1i = mchain_prev if bi > 0 else cv1_out_keep.get(i, even_keep(sb1.conv.in_channels, mb1.conv.in_channels))[:mb1.conv.in_channels]
                k2o = even_keep(sb2.conv.out_channels, mb2.conv.out_channels)
                if len(k1i) == mb1.conv.in_channels and len(k1o) == mb1.conv.out_channels:
                    mb1.conv.weight.data.copy_(sb1.conv.weight.data[k1o][:, k1i])
                    transferred += 1
                    for attr in ('weight', 'bias', 'running_mean', 'running_var'):
                        getattr(mb1.bn, attr).data.copy_(getattr(sb1.bn, attr).data[k1o])
                if len(k2o) == mb2.conv.out_channels and len(k1o) == mb2.conv.in_channels:
                    mb2.conv.weight.data.copy_(sb2.conv.weight.data[k2o][:, k1o])
                    transferred += 1
                    for attr in ('weight', 'bias', 'running_mean', 'running_var'):
                        getattr(mb2.bn, attr).data.copy_(getattr(sb2.bn, attr).data[k2o])
                mchain_prev = k2o
        # cv2: 输入是 concat 多段，段宽与 s 比例一致 (mid_w/s_w=0.75)，按段均匀采样
        s_cv2, m_cv2 = getattr(s_mod, 'cv2', None), getattr(m_mod, 'cv2', None)
        if s_cv2 is not None and m_cv2 is not None and type(s_mod).__name__ in ('C3k2', 'C2f'):
            k_o = even_keep(s_cv2.conv.out_channels, m_cv2.conv.out_channels)
            # s 语义的输入段: [前半 chunk (half_s) | m 链输出...]
            half_s = s_cv1.conv.out_channels // 2
            half_idx = even_keep(half_s, m_cv1.conv.out_channels // 2)
            # m 链最后一个 Bottleneck 的输出选择
            s_last_b = s_mod.m[-1]
            m_last_b = m_mod.m[-1]
            if not hasattr(s_last_b, 'cv2'):  # C2PSA 的 m 是 Sequential(PSABlock)，结构不同
                mchain_keep = None
            else:
                mchain_keep = even_keep(s_last_b.cv2.conv.out_channels, m_last_b.cv2.conv.out_channels)
            if mchain_keep is None:
                k_i = even_keep(s_cv2.conv.in_channels, m_cv2.conv.in_channels)
            else:
                k_i = half_idx + [h + half_s for h in mchain_keep]
            if len(k_o) == m_cv2.conv.out_channels and len(k_i) == m_cv2.conv.in_channels:
                m_cv2.conv.weight.data.copy_(s_cv2.conv.weight.data[k_o][:, k_i])
                transferred += 1
                for attr in ('weight', 'bias', 'running_mean', 'running_var'):
                    getattr(m_cv2.bn, attr).data.copy_(getattr(s_cv2.bn, attr).data[k_o])

    # Detect head: cv2/cv3 分支，输入通道来自 3 个 neck 输出
    s_det, m_det = src_seq[-1], mid_seq[-1]
    for attr in ('cv2', 'cv3', 'one2one_cv2', 'one2one_cv3'):
        s_branches, m_branches = getattr(s_det, attr, None), getattr(m_det, attr, None)
        if s_branches is None or m_branches is None:
            continue
        for bi, (s_seqb, m_seqb) in enumerate(zip(s_branches, m_branches)):
            # 分支可能是扁平 Sequential(Conv,Conv,Conv2d) 或嵌套 Sequential(Sequential(Conv,Conv),Conv2d)
            # 统一摊平成 conv 单元列表（每个单元是 Conv 包装或裸 Conv2d）
            def flatten_units(seqb):
                units = []
                for child in seqb.children():
                    if isinstance(child, nn.Sequential):
                        units.extend(flatten_units(child))
                    else:
                        units.append(child)
                return units

            neck_idx = [16, 19, 22][bi]
            k_in = out_keep.get(neck_idx)
            prev_keep = k_in
            for s_c, m_c in zip(flatten_units(s_seqb), flatten_units(m_seqb)):
                s_conv = s_c.conv if hasattr(s_c, 'conv') else s_c
                m_conv = m_c.conv if hasattr(m_c, 'conv') else m_c
                k_out = even_keep(s_conv.out_channels, m_conv.out_channels)
                is_dw = s_conv.groups > 1
                # DWConv(groups>1): 输入通道=输出通道，只切输出维；普通卷积切输入+输出
                in_ok = (len(k_out) == m_conv.in_channels) if is_dw else (
                    prev_keep is not None and len(prev_keep) == m_conv.in_channels)
                if in_ok and len(k_out) == m_conv.out_channels:
                    w = s_conv.weight.data[k_out]
                    if not is_dw:
                        w = w[:, prev_keep]
                    m_conv.weight.data.copy_(w)
                    transferred += 1
                    s_bn = getattr(s_c, 'bn', None)
                    m_bn = getattr(m_c, 'bn', None)
                    if isinstance(s_bn, nn.BatchNorm2d) and isinstance(m_bn, nn.BatchNorm2d):
                        for a in ('weight', 'bias', 'running_mean', 'running_var'):
                            getattr(m_bn, a).data.copy_(getattr(s_bn, a).data[k_out])
                prev_keep = k_out

    n = sum(p.numel() for p in mid_model.parameters())
    print(f'transferred tensors: {transferred}, skipped: {len(skipped)}')
    for s in skipped[:8]:
        print('  skip:', s)
    print(f'mid model: {n/1e6:.2f}M params')

    # 前向 sanity
    with torch.no_grad():
        y = mid_model(torch.randn(1, 3, 640, 640))
    print('forward ok:', type(y).__name__)

    ckpt = {'model': mid_model, 'train_args': {}}
    torch.save(ckpt, OUT)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
