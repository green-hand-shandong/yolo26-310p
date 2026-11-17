"""GridSample 融合拆解 v2：单次 batched GatherElements + ReduceSum

v1 的问题：每 GridSample 产生 ~100 个小算子（4 次 GE + 逐邻居 mask + 逐权重 Mul），
8.2 ATC 不会融合它们，导致 289ms。

v2 融合策略：
  1. 四邻索引拼成一个 (B, NR*4, C) 索引张量 → **单次 GatherElements** 取全部 4 邻
  2. 权重 (B, NR*4, 1) → 单次 Mul（广播 C）
  3. Reshape (B, NR, 4, C) → **单次 ReduceSum**（axis=2）完成四邻加权求和
  每 GridSample 从 ~100 节点降到 ~25 节点，且 GatherElements 只 1 次。
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnx.shape_inference as si


def make_gridsample_fused(node, vi_map, prefix):
    x_name, grid_name, out_name = node.input[0], node.input[1], node.output[0]
    x_shape = [d.dim_value for d in vi_map[x_name].type.tensor_type.shape.dim]
    g_shape = [d.dim_value for d in vi_map[grid_name].type.tensor_type.shape.dim]
    B, C, H, W = x_shape
    _, N, R, _ = g_shape
    NR = N * R

    inits, nodes = [], []

    def T(name, dtype, dims, vals):
        inits.append(helper.make_tensor(name, dtype, dims, vals))
        return name

    # ---- grid (B,N,R,2) -> gx (B,NR), gy (B,NR)
    sp_in = f'{prefix}.sp'
    inits.append(helper.make_tensor(sp_in, TensorProto.INT64, [2], [1, 1]))
    gx4 = f'{prefix}.gx4'
    gy4 = f'{prefix}.gy4'
    nodes.append(helper.make_node('Split', [grid_name, sp_in], [gx4, gy4], axis=3, name=f'{prefix}.splitxy'))
    rs2 = T(f'{prefix}.rs2', TensorProto.INT64, [2], [B, NR])
    gx = f'{prefix}.gx'
    gy = f'{prefix}.gy'
    nodes.append(helper.make_node('Reshape', [gx4, rs2], [gx], name=f'{prefix}.rsx2'))
    nodes.append(helper.make_node('Reshape', [gy4, rs2], [gy], name=f'{prefix}.rsy2'))

    # ---- unnormalize (align_corners=0): p = (g+1)*size/2 - 0.5
    def unnorm(g, size, tag):
        one = T(f'{prefix}.one{tag}', TensorProto.FLOAT, [1], [1.0])
        half = T(f'{prefix}.half{tag}', TensorProto.FLOAT, [1], [np.float32(size / 2)])
        m05 = T(f'{prefix}.m05{tag}', TensorProto.FLOAT, [1], [-0.5])
        p1 = f'{prefix}.p1{tag}'
        nodes.append(helper.make_node('Add', [g, one], [p1], name=f'{prefix}.u1{tag}'))
        pm = f'{prefix}.pm{tag}'
        nodes.append(helper.make_node('Mul', [p1, half], [pm], name=f'{prefix}.u2{tag}'))
        p = f'{prefix}.p{tag}'
        nodes.append(helper.make_node('Add', [pm, m05], [p], name=f'{prefix}.u3{tag}'))
        return p

    px = unnorm(gx, W, 'x')
    py = unnorm(gy, H, 'y')

    # ---- floor + fractional weights
    def floor_w(p, tag):
        fl = f'{prefix}.fl{tag}'
        nodes.append(helper.make_node('Floor', [p], [fl], name=f'{prefix}.fl{tag}'))
        w1 = f'{prefix}.w1{tag}'
        nodes.append(helper.make_node('Sub', [p, fl], [w1], name=f'{prefix}.w1{tag}'))
        one = T(f'{prefix}.one{tag}b', TensorProto.FLOAT, [1], [1.0])
        w0 = f'{prefix}.w0{tag}'
        nodes.append(helper.make_node('Sub', [one, w1], [w0], name=f'{prefix}.w0{tag}'))
        return fl, w0, w1

    fx, wx0, wx1 = floor_w(px, 'x')
    fy, wy0, wy1 = floor_w(py, 'y')

    # ---- clamp 索引（保证访存安全）
    def clamp(f, size, tag):
        fi = f'{prefix}.fi{tag}'
        nodes.append(helper.make_node('Cast', [f], [fi], to=TensorProto.INT64, name=f'{prefix}.c{tag}'))
        z = T(f'{prefix}.z{tag}', TensorProto.INT64, [1], [0])
        mx = T(f'{prefix}.mx{tag}', TensorProto.INT64, [1], [size - 1])
        c0 = f'{prefix}.c0{tag}'
        nodes.append(helper.make_node('Max', [fi, z], [c0], name=f'{prefix}.mx1{tag}'))
        c1 = f'{prefix}.c1{tag}'
        nodes.append(helper.make_node('Min', [c0, mx], [c1], name=f'{prefix}.mn1{tag}'))
        return c1

    xi = clamp(fx, W, 'x')
    yi = clamp(fy, H, 'y')

    # ---- 每邻居合法 mask（0<=p<size；zeros 语义 = 越界邻居贡献 0）
    # 邻居合法 mask（显式上下界）:
    #   x0=floor(px) 合法 ⟺ 0 <= px < W;   x1=x0+1 合法 ⟺ 0 <= px < W-1
    #   y0=floor(py) 合法 ⟺ 0 <= py < H;   y1=y0+1 合法 ⟺ 0 <= py < H-1
    def legal(p, up, tag):
        z = T(f'{prefix}.lz{tag}', TensorProto.FLOAT, [1], [0.0])
        mx = T(f'{prefix}.lm{tag}', TensorProto.FLOAT, [1], [np.float32(up)])
        lo = f'{prefix}.lo{tag}'
        nodes.append(helper.make_node('Greater', [p, z], [lo], name=f'{prefix}.lo{tag}'))
        hi = f'{prefix}.hi{tag}'
        nodes.append(helper.make_node('Less', [p, mx], [hi], name=f'{prefix}.hi{tag}'))
        a = f'{prefix}.an{tag}'
        nodes.append(helper.make_node('And', [lo, hi], [a], name=f'{prefix}.an{tag}'))
        af = f'{prefix}.af{tag}'
        nodes.append(helper.make_node('Cast', [a], [af], to=TensorProto.FLOAT, name=f'{prefix}.af{tag}'))
        return af

    l_x0 = legal(px, W, 'x0')          # px < W
    l_x1 = legal(px, W - 1, 'x1')      # px < W-1
    l_y0 = legal(py, H, 'y0')          # py < H
    l_y1 = legal(py, H - 1, 'y1')      # py < H-1

    # ---- 四邻 2D 展平索引（含 +1 未 clamp 邻居，用 clamp 保证安全 + mask 清零贡献）
    wT = T(f'{prefix}.wT', TensorProto.INT64, [1], [W])
    one64 = T(f'{prefix}.o64', TensorProto.INT64, [1], [1])
    y1i = f'{prefix}.y1i'
    nodes.append(helper.make_node('Add', [yi, one64], [y1i], name=f'{prefix}.y1a'))
    x1i = f'{prefix}.x1i'
    nodes.append(helper.make_node('Add', [xi, one64], [x1i], name=f'{prefix}.x1a'))
    # clamp +1 邻居
    mxH = T(f'{prefix}.mxH', TensorProto.INT64, [1], [H - 1])
    mxW = T(f'{prefix}.mxW', TensorProto.INT64, [1], [W - 1])
    y1c = f'{prefix}.y1c'
    nodes.append(helper.make_node('Min', [y1i, mxH], [y1c], name=f'{prefix}.y1c'))
    x1c = f'{prefix}.x1c'
    nodes.append(helper.make_node('Min', [x1i, mxW], [x1c], name=f'{prefix}.x1c'))

    # 4 邻索引 (B,NR): (y0,x0) (y0,x1) (y1,x0) (y1,x1)
    def idx2d(yy, xx, tag):
        a = f'{prefix}.ia{tag}'
        nodes.append(helper.make_node('Mul', [yy, wT], [a], name=f'{prefix}.im{tag}'))
        b = f'{prefix}.ib{tag}'
        nodes.append(helper.make_node('Add', [a, xx], [b], name=f'{prefix}.ia{tag}'))
        return b

    i00 = idx2d(yi, xi, '00')
    i01 = idx2d(yi, x1c, '01')
    i10 = idx2d(y1c, xi, '10')
    i11 = idx2d(y1c, x1c, '11')

    # Concat 成 (B, NR, 4) → Unsqueeze (B,NR,4,1) → Mul ones → (B,NR,4,C)
    i00u = f'{prefix}.i00u'
    i01u = f'{prefix}.i01u'
    i10u = f'{prefix}.i10u'
    i11u = f'{prefix}.i11u'
    unsq2 = T(f'{prefix}.unsq2', TensorProto.INT64, [1], [2])
    nodes.append(helper.make_node('Unsqueeze', [i00, unsq2], [i00u], name=f'{prefix}.u00'))
    nodes.append(helper.make_node('Unsqueeze', [i01, unsq2], [i01u], name=f'{prefix}.u01'))
    nodes.append(helper.make_node('Unsqueeze', [i10, unsq2], [i10u], name=f'{prefix}.u10'))
    nodes.append(helper.make_node('Unsqueeze', [i11, unsq2], [i11u], name=f'{prefix}.u11'))
    cat = f'{prefix}.cat'
    nodes.append(helper.make_node('Concat', [i00u, i01u, i10u, i11u], [cat], axis=2, name=f'{prefix}.cat'))
    unsq2b = T(f'{prefix}.unsq2b', TensorProto.INT64, [1], [3])
    b4 = f'{prefix}.b4'
    nodes.append(helper.make_node('Unsqueeze', [cat, unsq2b], [b4], name=f'{prefix}.b4'))  # (B,NR,4,1)
    onesC = T(f'{prefix}.onesC', TensorProto.INT64, [1, 1, 1, C], [1] * C)
    idxAll = f'{prefix}.idxAll'
    nodes.append(helper.make_node('Mul', [b4, onesC], [idxAll], name=f'{prefix}.exp'))  # (B,NR,4,C)

    # ---- 特征图展平 (B, HW, C)
    perm = f'{prefix}.perm'
    inits.append(helper.make_tensor(perm, TensorProto.INT64, [4], [0, 2, 3, 1]))
    xt = f'{prefix}.xt'
    nodes.append(helper.make_node('Transpose', [x_name], [xt], perm=[0, 2, 3, 1], name=f'{prefix}.tr'))
    hw = T(f'{prefix}.hw', TensorProto.INT64, [3], [B, H * W, C])
    xf = f'{prefix}.xf'
    nodes.append(helper.make_node('Reshape', [xt, hw], [xf], name=f'{prefix}.rsx'))

    # ---- 单次 GatherElements: (B, NR*4, C)
    rs4 = T(f'{prefix}.rs4', TensorProto.INT64, [3], [B, NR * 4, C])
    idxFlat = f'{prefix}.idxFlat'
    nodes.append(helper.make_node('Reshape', [idxAll, rs4], [idxFlat], name=f'{prefix}.rsi'))
    gathered = f'{prefix}.gathered'
    nodes.append(helper.make_node('GatherElements', [xf, idxFlat], [gathered], axis=1, name=f'{prefix}.ge'))

    # ---- 权重 × 邻居合法 mask（zeros padding: 越界邻居贡献 0）
    def wmul2(a, b, tag):
        o = f'{prefix}.wm{tag}'
        nodes.append(helper.make_node('Mul', [a, b], [o], name=f'{prefix}.wm{tag}'))
        return o

    w00 = wmul2(wmul2(wmul2(wy0, wx0, 'w00a'), l_x0, 'w00b'), l_y0, 'w00c')
    w01 = wmul2(wmul2(wmul2(wy0, wx1, 'w01a'), l_x1, 'w01b'), l_y0, 'w01c')
    w10 = wmul2(wmul2(wmul2(wy1, wx0, 'w10a'), l_x0, 'w10b'), l_y1, 'w10c')
    w11 = wmul2(wmul2(wmul2(wy1, wx1, 'w11a'), l_x1, 'w11b'), l_y1, 'w11c')

    # ---- 权重: w00..w11 (B,NR) → Unsqueeze 各自 (B,NR,1) → Concat (B,NR,4) → Reshape (B,NR*4,1)
    unsq3 = T(f'{prefix}.unsq3', TensorProto.INT64, [1], [2])
    def unsq_w(w, tag):
        o = f'{prefix}.wu{tag}'
        nodes.append(helper.make_node('Unsqueeze', [w, unsq3], [o], name=f'{prefix}.wu{tag}'))
        return o
    w00u = unsq_w(w00, '00')
    w01u = unsq_w(w01, '01')
    w10u = unsq_w(w10, '10')
    w11u = unsq_w(w11, '11')
    wcat = f'{prefix}.wcat'
    nodes.append(helper.make_node('Concat', [w00u, w01u, w10u, w11u], [wcat], axis=2, name=f'{prefix}.wcat'))
    wrs = f'{prefix}.wrs'
    wf = f'{prefix}.wf'
    inits.append(helper.make_tensor(f'{prefix}.wshape', TensorProto.INT64, [3], [B, NR * 4, 1]))
    nodes.append(helper.make_node('Reshape', [wcat, f'{prefix}.wshape'], [wf], name=f'{prefix}.wrs'))

    # ---- 加权: gathered (B,NR*4,C) * wf (B,NR*4,1) → 广播
    weighted = f'{prefix}.weighted'
    nodes.append(helper.make_node('Mul', [gathered, wf], [weighted], name=f'{prefix}.wm'))

    # ---- Reshape (B, NR, 4, C) → ReduceSum axis=2 → (B, NR, C)
    rsN4 = T(f'{prefix}.rsN4', TensorProto.INT64, [4], [B, NR, 4, C])
    wsum = f'{prefix}.wsum'
    nodes.append(helper.make_node('Reshape', [weighted, rsN4], [wsum], name=f'{prefix}.rs4'))
    red = f'{prefix}.red'
    axes2 = T(f'{prefix}.axes2', TensorProto.INT64, [1], [2])
    nodes.append(helper.make_node('ReduceSum', [wsum, axes2], [red], keepdims=0, name=f'{prefix}.reduce'))

    # ---- 输出 reshape (B,N,R,C) → transpose (B,C,N,R)
    rsO = T(f'{prefix}.rsO', TensorProto.INT64, [4], [B, N, R, C])
    resh = f'{prefix}.resh'
    nodes.append(helper.make_node('Reshape', [red, rsO], [resh], name=f'{prefix}.rsO'))
    nodes.append(helper.make_node('Transpose', [resh], [out_name], perm=[0, 3, 1, 2], name=f'{prefix}.trO'))
    return nodes, inits


def replace_gridsample_fused(onnx_path, out_path):
    m = onnx.load(onnx_path)
    inf = si.infer_shapes(m)
    vi_map = {v.name: v for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)}
    new_nodes, all_inits = [], []
    cnt = 0
    for n in m.graph.node:
        if n.op_type == 'GridSample':
            cnt += 1
            nodes, inits = make_gridsample_fused(n, vi_map, f'gs{cnt}')
            new_nodes.extend(nodes)
            all_inits.extend(inits)
        else:
            new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    m.graph.initializer.extend(all_inits)
    onnx.save(m, out_path)
    print(f'replaced {cnt} GridSample with fused version -> {out_path}')


if __name__ == '__main__':
    replace_gridsample_fused('/data2/gxw/yolo26-310p-eval/rtdetr-l.onnx',
                             '/data2/gxw/yolo26-310p-eval/rtdetr-l_gsf.onnx')
