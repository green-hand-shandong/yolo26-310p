"""GridSample (bilinear, align_corners=0, padding_mode=zeros) 的算子级拆解

输入: x (B,C,H,W), grid (B,N,4,2)  [grid 第 3 维按 batch 展平处理]
输出: out (B,C,N,4) = x 采样于 grid 坐标处

等效实现（全部用 310P 已支持算子）:
  1. unnormalize: gx = (grid_x + 1) * W/2 - 0.5;  gy 同理   [Add/Div... 用 Mul+Add]
  2. 四邻坐标: x0=floor(gx), x1=x0+1, 并 clamp 到 [0, W-1]  [Floor/Add/Max/Min, Cast]
  3. 权重: wx1 = gx - x0f, wx0 = 1 - wx1（用未 clamp 的 floor 保持边界外语义）
  4. 采样: 用 GatherElements 沿 W 轴采 x0/x1（把 (B,C,H,W) 转置为 (B,W,C,H)?）
     —— GatherElements 需 index 与 data 维度一致。做法:
        x_b (B,C,H,W) → 按 H 先 Gather（index (B,1,N,H)?）复杂。
     更简单: 把 W 方向采样拆成两次 GatherElements with broadcasted index。
  5. zeros padding: clamp 索引 + mask（越界位置乘 0 权重 → 等效 zeros padding:
     对越界坐标: 用 clamp 后取值 * 边界mask。padding_mode=zeros 语义 = 越界处取 0。
     实现: valid = (gx>=0) & (gx<=W-1) 转成 0/1 mask 乘进权重。

本脚本: 遍历 ONNX 图，将每个 GridSample 节点替换为上述算子组合。
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnx.shape_inference as si


def make_gridsample_replacement(node, vi_map, prefix):
    """为一个 GridSample 节点生成等效子图（节点列表 + 新 initializer）"""
    x_name, grid_name, out_name = node.input[0], node.input[1], node.output[0]
    x_shape = [d.dim_value for d in vi_map[x_name].type.tensor_type.shape.dim]      # B,C,H,W
    g_shape = [d.dim_value for d in vi_map[grid_name].type.tensor_type.shape.dim]   # B,N,4,2
    B, C, H, W = x_shape
    _, N, R, _ = g_shape   # R=4 (multi-ref)

    inits = []
    nodes = []
    # ---- 展平 grid: (B,N,R,2) -> (B, N*R) 每个点的 (gx, gy)
    # grid: [B,N,R,2]; 取 gx = grid[...,0], gy = grid[...,1]
    # 用 Split 沿最后一维
    sp_in = f'{prefix}.sp2'
    inits.append(helper.make_tensor(sp_in, TensorProto.INT64, [2], [1, 1]))
    gx_flat = f'{prefix}.gx'   # (B,N,R,1)
    gy_flat = f'{prefix}.gy'
    nodes.append(helper.make_node('Split', [grid_name, sp_in], [gx_flat, gy_flat], axis=3, name=f'{prefix}.splitxy'))
    # reshape 到 (B, N*R)
    rs_shape = f'{prefix}.rs_shape'
    inits.append(helper.make_tensor(rs_shape, TensorProto.INT64, [2], [B, N * R]))
    gx = f'{prefix}.gxf'   # (B, NR)
    gy = f'{prefix}.gyf'
    nodes.append(helper.make_node('Reshape', [gx_flat, rs_shape], [gx], name=f'{prefix}.rs_gx'))
    nodes.append(helper.make_node('Reshape', [gy_flat, rs_shape], [gy], name=f'{prefix}.rs_gy'))

    # ---- unnormalize (align_corners=0): px = (g+1)*size/2 - 0.5
    def unnorm(g, size, tag):
        # align_corners=0: px = (g + 1) * size/2 - 0.5
        half = np.float32(size / 2.0)
        m1 = np.float32(-0.5)
        inits.append(helper.make_tensor(f'{prefix}.half{tag}', TensorProto.FLOAT, [1], [half]))
        inits.append(helper.make_tensor(f'{prefix}.m05{tag}', TensorProto.FLOAT, [1], [m1]))
        one = f'{prefix}.one{tag}'
        if not any(i.name == one for i in inits):
            inits.append(helper.make_tensor(one, TensorProto.FLOAT, [1], [1.0]))
        p1 = f'{prefix}.p1{tag}'
        nodes.append(helper.make_node('Add', [g, one], [p1], name=f'{prefix}.unnorm1{tag}'))
        pm = f'{prefix}.pm{tag}'
        nodes.append(helper.make_node('Mul', [p1, f'{prefix}.half{tag}'], [pm], name=f'{prefix}.unnorm2{tag}'))
        px = f'{prefix}.px{tag}'
        nodes.append(helper.make_node('Add', [pm, f'{prefix}.m05{tag}'], [px], name=f'{prefix}.unnorm3{tag}'))
        return px

    px = unnorm(gx, W, 'x')   # (B, NR) float
    py = unnorm(gy, H, 'y')

    # ---- floor 与权重
    def floor_w(px, size, tag):
        f = f'{prefix}.floor{tag}'
        nodes.append(helper.make_node('Floor', [px], [f], name=f'{prefix}.fl{tag}'))
        # 权重 w1 = px - floor  (0..1)
        w1 = f'{prefix}.w1{tag}'
        nodes.append(helper.make_node('Sub', [px, f], [w1], name=f'{prefix}.w1{tag}'))
        w0 = f'{prefix}.w0{tag}'
        one = f'{prefix}.one{tag}'
        if not any(i.name == one for i in inits):
            inits.append(helper.make_tensor(one, TensorProto.FLOAT, [1], [1.0]))
        nodes.append(helper.make_node('Sub', [f'{prefix}.one{tag}', w1], [w0], name=f'{prefix}.w0{tag}'))
        return f, w0, w1

    fx, wx0, wx1 = floor_w(px, W, 'x')
    fy, wy0, wy1 = floor_w(py, H, 'y')

    # ---- clamp 索引到 [0, size-1]
    def clamp_idx(f, size, tag):
        fi = f'{prefix}.fi{tag}'
        nodes.append(helper.make_node('Cast', [f], [fi], to=TensorProto.INT64, name=f'{prefix}.cast{tag}'))
        z = f'{prefix}.z{tag}'
        mx = f'{prefix}.mx{tag}'
        inits.append(helper.make_tensor(z, TensorProto.INT64, [1], [0]))
        inits.append(helper.make_tensor(mx, TensorProto.INT64, [1], [size - 1]))
        c0n = f'{prefix}.cl0{tag}'
        nodes.append(helper.make_node('Max', [fi, z], [c0n], name=f'{prefix}.max{tag}'))
        c1n = f'{prefix}.cli{tag}'
        nodes.append(helper.make_node('Min', [c0n, mx], [c1n], name=f'{prefix}.min{tag}'))
        return c1n

    xi = clamp_idx(fx, W, 'x')   # (B, NR) int64
    yi = clamp_idx(fy, H, 'y')

    # ---- 邻居合法性 mask（padding_mode=zeros: 越界邻居贡献 0）
    # x0=floor(px) 合法: -1 < px < W;  x1=x0+1 合法: -1 < px < W-1
    # y0=floor(py) 合法: -1 < py < H;  y1=y0+1 合法: -1 < py < H-1
    # x0=floor(px) 合法 ⟺ 0 <= px < W;  x1=x0+1 合法 ⟺ 0 <= px < W-1
    z0x = f'{prefix}.z0x'
    inits.append(helper.make_tensor(z0x, TensorProto.FLOAT, [1], [0.0]))
    wx0_ok = f'{prefix}.wx0ok'
    nodes.append(helper.make_node('Greater', [px, z0x], [wx0_ok], name=f'{prefix}.g_x0a'))
    wx0_ok2 = f'{prefix}.wx0ok2'
    inits.append(helper.make_tensor(wx0_ok2 + '_v', TensorProto.FLOAT, [1], [np.float32(W)]))
    nodes.append(helper.make_node('Less', [px, wx0_ok2 + '_v'], [wx0_ok2], name=f'{prefix}.g_x0b'))
    wx0ok = f'{prefix}.wx0okc'
    nodes.append(helper.make_node('And', [wx0_ok, wx0_ok2], [wx0ok], name=f'{prefix}.a_x0'))
    wx1_ok = f'{prefix}.wx1ok'
    nodes.append(helper.make_node('Greater', [px, z0x], [wx1_ok], name=f'{prefix}.g_x1a'))
    wx1_ok2 = f'{prefix}.wx1ok2'
    inits.append(helper.make_tensor(wx1_ok2 + '_v', TensorProto.FLOAT, [1], [np.float32(W - 1)]))
    nodes.append(helper.make_node('Less', [px, wx1_ok2 + '_v'], [wx1_ok2], name=f'{prefix}.g_x1b'))
    nodes.append(helper.make_node('And', [f'{prefix}.wx1ok', wx1_ok2], [f'{prefix}.wx1okd'], name=f'{prefix}.a_x1'))
    wy0_ok = f'{prefix}.wy0ok'
    nodes.append(helper.make_node('Greater', [py, z0x], [wy0_ok], name=f'{prefix}.g_y0a'))
    wy0_ok2 = f'{prefix}.wy0ok2'
    inits.append(helper.make_tensor(wy0_ok2 + '_v', TensorProto.FLOAT, [1], [np.float32(H)]))
    nodes.append(helper.make_node('Less', [py, wy0_ok2 + '_v'], [wy0_ok2], name=f'{prefix}.g_y0b'))
    wy0ok = f'{prefix}.wy0okc'
    nodes.append(helper.make_node('And', [wy0_ok, wy0_ok2], [wy0ok], name=f'{prefix}.a_y0'))
    wy1_ok = f'{prefix}.wy1ok'
    nodes.append(helper.make_node('Greater', [py, z0x], [wy1_ok], name=f'{prefix}.g_y1a'))
    wy1_ok2 = f'{prefix}.wy1ok2'
    inits.append(helper.make_tensor(wy1_ok2 + '_v', TensorProto.FLOAT, [1], [np.float32(H - 1)]))
    nodes.append(helper.make_node('Less', [py, wy1_ok2 + '_v'], [wy1_ok2], name=f'{prefix}.g_y1b'))
    nodes.append(helper.make_node('And', [f'{prefix}.wy1ok', wy1_ok2], [f'{prefix}.wy1okd'], name=f'{prefix}.a_y1'))

    def bcast(b, tag):
        o = f'{prefix}.bc{tag}'
        nodes.append(helper.make_node('Cast', [b], [o], to=TensorProto.FLOAT, name=f'{prefix}.c{tag}'))
        return o

    lx0b = bcast(wx0ok, 'x0')
    ly0b = bcast(wy0ok, 'y0')
    lxsb = bcast(f'{prefix}.wx1okd', 'x1')
    lysb = bcast(f'{prefix}.wy1okd', 'y1')
    def wmul(a, b, tag):
        o = f'{prefix}.wm{tag}'
        nodes.append(helper.make_node('Mul', [a, b], [o], name=f'{prefix}.wm{tag}'))
        return o

    def wadd(a, b, tag):
        o = f'{prefix}.wa{tag}'
        nodes.append(helper.make_node('Add', [a, b], [o], name=f'{prefix}.wa{tag}'))
        return o

    def smul(s, w, tag):
        o = f'{prefix}.sm{tag}'
        nodes.append(helper.make_node('Mul', [s, w], [o], name=f'{prefix}.sm{tag}'))
        return o

    # ---- 展平特征图并构造 2D 索引
    perm = f'{prefix}.perm'
    inits.append(helper.make_tensor(perm, TensorProto.INT64, [4], [0, 2, 3, 1]))
    xt = f'{prefix}.xt'
    nodes.append(helper.make_node('Transpose', [x_name], [xt], perm=[0, 2, 3, 1], name=f'{prefix}.tr'))
    hw = f'{prefix}.hw'
    inits.append(helper.make_tensor(hw, TensorProto.INT64, [3], [B, H * W, C]))
    xf = f'{prefix}.xf'
    nodes.append(helper.make_node('Reshape', [xt, hw], [xf], name=f'{prefix}.rsx'))  # (B, HW, C)

    wT = f'{prefix}.wT'
    inits.append(helper.make_tensor(wT, TensorProto.INT64, [1], [W]))
    yw = f'{prefix}.yw'
    nodes.append(helper.make_node('Mul', [yi, wT], [yw], name=f'{prefix}.yw'))
    idx2d = f'{prefix}.idx2d'
    nodes.append(helper.make_node('Add', [yw, xi], [idx2d], name=f'{prefix}.i2d'))

    onesi = f'{prefix}.onesi'
    inits.append(helper.make_tensor(onesi, TensorProto.INT64, [1, 1, C], [1] * C))
    b3 = f'{prefix}.b3'
    axes2 = f'{prefix}.axes2'
    inits.append(helper.make_tensor(axes2, TensorProto.INT64, [1], [2]))
    nodes.append(helper.make_node('Unsqueeze', [idx2d, axes2], [b3], name=f'{prefix}.b3'))
    idxX = f'{prefix}.idxX'
    nodes.append(helper.make_node('Mul', [b3, onesi], [idxX], name=f'{prefix}.exp'))

    one64 = f'{prefix}.one64'
    inits.append(helper.make_tensor(one64, TensorProto.INT64, [1], [1]))
    yi1r = f'{prefix}.yi1r'
    nodes.append(helper.make_node('Add', [yi, one64], [yi1r], name=f'{prefix}.yi1a'))
    xi1r = f'{prefix}.xi1r'
    nodes.append(helper.make_node('Add', [xi, one64], [xi1r], name=f'{prefix}.xi1a'))
    yi1c = f'{prefix}.yi1c'
    inits.append(helper.make_tensor(f'{prefix}.mxH', TensorProto.INT64, [1], [H - 1]))
    nodes.append(helper.make_node('Min', [yi1r, f'{prefix}.mxH'], [yi1c], name=f'{prefix}.yi1c'))
    xi1c = f'{prefix}.xi1c'
    inits.append(helper.make_tensor(f'{prefix}.mxW', TensorProto.INT64, [1], [W - 1]))
    nodes.append(helper.make_node('Min', [xi1r, f'{prefix}.mxW'], [xi1c], name=f'{prefix}.xi1c'))

    def idx2d_from(yy, xx, tag):
        a = f'{prefix}.a{tag}w'
        nodes.append(helper.make_node('Mul', [yy, wT], [a], name=a))
        b = f'{prefix}.a{tag}s'
        nodes.append(helper.make_node('Add', [a, xx], [b], name=b))
        b3 = f'{prefix}.a{tag}b3'
        nodes.append(helper.make_node('Unsqueeze', [b, axes2], [b3], name=b + '_u'))
        e = f'{prefix}.a{tag}e'
        nodes.append(helper.make_node('Mul', [b3, onesi], [e], name=e))
        return e

    idx00 = idx2d_from(yi, xi, '00')
    idx01 = idx2d_from(yi, xi1c, '01')
    idx10 = idx2d_from(yi1c, xi, '10')
    idx11 = idx2d_from(yi1c, xi1c, '11')

    def sample(idx, tag):
        ix = f'{prefix}.s{tag}'
        nodes.append(helper.make_node('GatherElements', [xf, idx], [ix], axis=1, name=f'{prefix}.ge{tag}'))
        return ix

    s00 = sample(idx00, '00')
    s01 = sample(idx01, '01')
    s10 = sample(idx10, '10')
    s11 = sample(idx11, '11')

    # 邻居 (row, col) 合法性: w00=(y0,x0), w01=(y0,x1), w10=(y1,x0), w11=(y1,x1)
    w00 = wmul(wmul(wmul(wy0, wx0, '00'), lx0b, 'lx0'), ly0b, 'ly0')
    w01 = wmul(wmul(wmul(wy0, wx1, '01'), lxsb, 'lxsb'), ly0b, 'ly0b')
    w10 = wmul(wmul(wmul(wy1, wx0, '10'), lx0b, 'lx0b'), lysb, 'lysb')
    w11 = wmul(wmul(wmul(wy1, wx1, '11'), lxsb, 'lxsb1'), lysb, 'lysb1')

    # 权重广播到 (B, NR, C): Expand with shape (1,1,C)
    def wb(w, tag):
        # Reshape (B,NR) -> (B,NR,1)，后续 Mul 依赖广播到 (B,NR,C)
        o = f'{prefix}.wb{tag}'
        rs = f'{prefix}.wbrs{tag}'
        inits.append(helper.make_tensor(rs, TensorProto.INT64, [3], [B, N * R, 1]))
        nodes.append(helper.make_node('Reshape', [w, rs], [o], name=f'{prefix}.wb{tag}'))
        return o

    w00b, w01b, w10b, w11b = wb(w00, '00'), wb(w01, '01'), wb(w10, '10'), wb(w11, '11')

    def wadd(a, b, tag):
        o = f'{prefix}.wa{tag}'
        nodes.append(helper.make_node('Add', [a, b], [o], name=f'{prefix}.wa{tag}'))
        return o

    wa01 = wadd(w00b, w01b, '01')
    wa23 = wadd(w10b, w11b, '23')
    wsum = wadd(wa01, wa23, 'all')

    # 加权求和: s00*w00 + s01*w01 + s10*w10 + s11*w11
    def smul(s, w, tag):
        o = f'{prefix}.sm{tag}'
        nodes.append(helper.make_node('Mul', [s, w], [o], name=f'{prefix}.sm{tag}'))
        return o

    p00 = smul(s00, w00b, '00')
    p01 = smul(s01, w01b, '01')
    p10 = smul(s10, w10b, '10')
    p11 = smul(s11, w11b, '11')
    a1 = wadd(p00, p01, 'a1')
    a2 = wadd(p10, p11, 'a2')
    total = wadd(a1, a2, 'tot')

    # 输出 (B, NR, C) → Reshape (B, N, R, C) → Transpose (B, C, N, R)
    rsN = f'{prefix}.rsN'
    inits.append(helper.make_tensor(rsN, TensorProto.INT64, [4], [B, N, R, C]))
    resh = f'{prefix}.resh'
    nodes.append(helper.make_node('Reshape', [total, rsN], [resh], name=f'{prefix}.rsO'))
    nodes.append(helper.make_node('Transpose', [resh], [out_name], perm=[0, 3, 1, 2], name=f'{prefix}.trO'))
    # 上面 total 未除以 wsum？双线性权重 (wy0+wy1=1)(wx0+wx1=1) 在内部点自动归一，边界 clamp 处
    # w0/w1 仍和为 1（clamp 只影响取值索引），zeros padding 用 vmask 乘 → 越界全 0 ✓
    return nodes, inits


def replace_gridsample(onnx_path, out_path):
    m = onnx.load(onnx_path)
    inf = si.infer_shapes(m)
    vi_map = {v.name: v for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)}
    new_nodes = []
    all_inits = []
    gs_count = 0
    for n in m.graph.node:
        if n.op_type == 'GridSample':
            gs_count += 1
            prefix = f'gs{gs_count}'
            nodes, inits = make_gridsample_replacement(n, vi_map, prefix)
            new_nodes.extend(nodes)
            all_inits.extend(inits)
        else:
            new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    m.graph.initializer.extend(all_inits)
    # 清理 opset: GridSample 属于 opset16 域；替换后不再需要
    onnx.save(m, out_path)
    print(f'replaced {gs_count} GridSample -> {out_path}')
    return gs_count


if __name__ == '__main__':
    n = replace_gridsample('/data2/gxw/yolo26-310p-eval/rtdetr-l.onnx',
                           '/data2/gxw/yolo26-310p-eval/rtdetr-l_gsd.onnx')
    # 数值对照（CPU ORT）
    import onnxruntime as ort
    x = np.random.rand(1, 3, 64, 64).astype(np.float32)  # 尺寸不重要，先验证可运行
    print('saved. next: verify numerics vs original on CPU')
