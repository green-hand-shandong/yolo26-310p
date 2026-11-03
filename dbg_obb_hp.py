#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB 重编译验证: high_performance 模式重编译后, 完整流程 + 数值比对"""
import acl
import numpy as np
import time

OM = "/work/yolo26n_obb_310p_hp.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc()
acl.mdl.get_desc(d, mid)
n_in = acl.mdl.get_num_inputs(d)
n_out = acl.mdl.get_num_outputs(d)
print(f"model: inputs={n_in} outputs={n_out}")

ds = acl.mdl.create_dataset()
np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
dev_in, r = acl.rt.malloc(np_in.nbytes, 0)
acl.rt.memcpy(dev_in, np_in.nbytes, acl.util.numpy_to_ptr(np_in), np_in.nbytes, 1)
r1 = acl.create_data_buffer(dev_in, np_in.nbytes)
acl.mdl.add_dataset_buffer(ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
dev_ph, r = acl.rt.malloc(8, 0)
r2 = acl.create_data_buffer(dev_ph, 8)
acl.mdl.add_dataset_buffer(ds, r2[0] if isinstance(r2, (tuple, list)) else r2)

out0_sz = 8400
dev_out, r = acl.rt.malloc(out0_sz, 0)
r3 = acl.create_data_buffer(dev_out, out0_sz)
acl.mdl.add_dataset_buffer(ds, r3[0] if isinstance(r3, (tuple, list)) else r3)

for _ in range(10):
    acl.mdl.execute(mid, ds, ds)

host = np.empty(2100, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), 8400, dev_out, 8400, 2)
out = host.reshape(1, 300, 7)
det = out[0][out[0][..., 4] >= 0.25]
print("conf>=0.25 检测数:", len(det))
for b in det[:8]:
    print(f"  score={b[4]:.3f} box={b[:4].astype(int)} angle={b[5]:.3f}")

# 与 fp32 基线比对
ref = np.load("/work/ref_obb_fp32.npy")
r_b = ref[0][ref[0][..., 4] >= 0.25]
print("fp32 基线检测数:", len(r_b))
if len(det) and len(r_b):
    def iou(a, b):
        x1,y1,x2,y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
        inter = max(0,x2-x1)*max(0,y2-y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua>0 else 0
    matched = 0
    for b310 in det:
        best = 0
        for rfp in r_b:
            v = iou(b310[:4], rfp[:4])
            if v > best: best = v
        if best > 0.5: matched += 1
    print(f"一致性: {matched}/{len(det)} 匹配 (IoU>0.5)")
np.save("/work/out_obb_310p_hp.npy", out)
print("OBB HP TEST DONE")
