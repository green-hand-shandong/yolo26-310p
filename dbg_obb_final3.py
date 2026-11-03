#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB 最终验证: 完全复刻 fix2 成功路径 —— dataset 只加 [in0, in_ph] 两个 buffer,
execute(ds, ds) 自引用, 然后 dump dev_in 头部 8400 字节作为输出。"""
import acl
import numpy as np
import time

OM = "/work/yolo26n_obb_310p.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc()
acl.mdl.get_desc(d, mid)

ds = acl.mdl.create_dataset()
np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
in_np_backup = np_in.copy()
dev_in, r = acl.rt.malloc(np_in.nbytes, 0)
acl.rt.memcpy(dev_in, np_in.nbytes, acl.util.numpy_to_ptr(np_in), np_in.nbytes, 1)
r1 = acl.create_data_buffer(dev_in, np_in.nbytes)
acl.mdl.add_dataset_buffer(ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
dev_ph, r = acl.rt.malloc(8, 0)
r2 = acl.create_data_buffer(dev_ph, 8)
acl.mdl.add_dataset_buffer(ds, r2[0] if isinstance(r2, (tuple, list)) else r2)

for _ in range(10):
    ret = acl.mdl.execute(mid, ds, ds)

# dump dev_in 头部 8400 字节
host_head = np.empty(2100, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host_head), 8400, dev_in, 8400, 2)
out = host_head.reshape(1, 300, 7)
det = out[0][out[0][..., 4] >= 0.25]
print("execute ret:", ret, "| conf>=0.25 检测数:", len(det))
for b in det[:8]:
    print(f"  score={b[4]:.3f} box={b[:4].astype(int)} angle={b[5]:.3f} cls={b[6]}")

# 验证输入是否被覆写
host_in = np.empty(np_in.nbytes // 4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host_in), np_in.nbytes, dev_in, np_in.nbytes, 2)
print("input intact:", np.allclose(host_in, in_np_backup, atol=1e-4))

np.save("/work/out_obb_310p_final.npy", out)

N = 100
lat = []
for _ in range(N):
    t0 = time.perf_counter()
    acl.mdl.execute(mid, ds, ds)
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
print(f"latency ms: avg={lat.mean():.2f} p50={np.percentile(lat,50):.2f} p99={np.percentile(lat,99):.2f}")
print("OBB FINAL DONE")
