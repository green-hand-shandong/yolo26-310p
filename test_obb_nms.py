#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB 正解验证: one2many(nms=True) OM, execute ret=0, 输出 box 像素量级正常。
完整流程 + fp32 基线比对 + 测速。"""
import acl
import numpy as np
import time

OM = "/work/yolo26n_obb_nms_310p_linux_aarch64.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc()
acl.mdl.get_desc(d, mid)

np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
in_ds = acl.mdl.create_dataset()
dev_in, r = acl.rt.malloc(np_in.nbytes, 0)
acl.rt.memcpy(dev_in, np_in.nbytes, acl.util.numpy_to_ptr(np_in), np_in.nbytes, 1)
r1 = acl.create_data_buffer(dev_in, np_in.nbytes)
acl.mdl.add_dataset_buffer(in_ds, r1[0] if isinstance(r1, (tuple, list)) else r1)

out_sizes = [acl.mdl.get_output_size_by_index(d, i) for i in range(acl.mdl.get_num_outputs(d))]
out_ds = acl.mdl.create_dataset()
dev_outs = []
for sz in out_sizes:
    dev_o, r = acl.rt.malloc(sz, 0)
    r2 = acl.create_data_buffer(dev_o, sz)
    acl.mdl.add_dataset_buffer(out_ds, r2[0] if isinstance(r2, (tuple, list)) else r2)
    dev_outs.append(dev_o)

for _ in range(10):
    ret = acl.mdl.execute(mid, in_ds, out_ds)
print("execute ret:", ret)

host = np.empty(out_sizes[0] // 4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_sizes[0], dev_outs[0], out_sizes[0], 2)
out = host.reshape(1, 300, 7)
np.save("/work/out_obb_nms_310p.npy", out)

# fp32 基线比对
ref = np.load("/work/ref_obb_fp32.npy")
b310, bfp = out[0], ref[0]
print("310P score col: min=%.4f max=%.4f" % (b310[:,4].min(), b310[:,4].max()))
print("fp32 score col: min=%.4f max=%.4f" % (bfp[:,4].min(), bfp[:,4].max()))
print("310P box col: %.1f~%.1f" % (b310[:,:4].min(), b310[:,:4].max()))
print("fp32 box col: %.1f~%.1f" % (bfp[:,:4].min(), bfp[:,:4].max()))
print("310P angle col: %.2f~%.2f" % (b310[:,6].min(), b310[:,6].max()))
print("fp32 angle col: %.2f~%.2f" % (bfp[:,6].min(), bfp[:,6].max()))
# 全 2100 值最大绝对差
print("最大绝对差: %.4f" % np.abs(b310 - bfp).max())

# 测速
N = 100
lat = []
for _ in range(N):
    t0 = time.perf_counter()
    acl.mdl.execute(mid, in_ds, out_ds)
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
print(f"latency ms: avg={lat.mean():.2f} p50={np.percentile(lat,50):.2f} p99={np.percentile(lat,99):.2f}")
print("OBB NMS TEST DONE")
