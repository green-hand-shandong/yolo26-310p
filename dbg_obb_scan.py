#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB 诊断收尾: dump dev_in 前 128KB 扫描 —— 找 box 量级(几十~几百)的数据段, 确定 out blob 真实偏移"""
import acl
import numpy as np

OM = "/work/yolo26n_obb_310p.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)

ds = acl.mdl.create_dataset()
np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
dev_in, r = acl.rt.malloc(np_in.nbytes, 0)
acl.rt.memcpy(dev_in, np_in.nbytes, acl.util.numpy_to_ptr(np_in), np_in.nbytes, 1)
r1 = acl.create_data_buffer(dev_in, np_in.nbytes)
acl.mdl.add_dataset_buffer(ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
dev_ph, r = acl.rt.malloc(8, 0)
r2 = acl.create_data_buffer(dev_ph, 8)
acl.mdl.add_dataset_buffer(ds, r2[0] if isinstance(r2, (tuple, list)) else r2)

for _ in range(3):
    acl.mdl.execute(mid, ds, ds)

# dump dev_in 前 1MB
SZ = 262144  # 64K floats
host = np.empty(SZ // 4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), SZ, dev_in, SZ, 2)
# 扫描: 找值域在 5~3000 的连续段 (box 坐标量级)
mask = (host > 5) & (host < 3000)
idx = np.where(mask)[0]
if len(idx):
    print("找到疑似 box 数据: offset(float)=%d~%d, 前10值=%s" % (idx[0], idx[-1], host[idx[:10]].round(2)))
else:
    print("1MB 内无 box 量级数据 → 输出 blob 不在 dev_in")
    # 检查 dev_ph 位置: dump dev_ph 前后
    host2 = np.empty(2048, dtype=np.float32)
    acl.rt.memcpy(acl.util.numpy_to_ptr(host2), 8192, dev_ph, 8192, 2)
    print("dev_ph 区域:", host2[:16].round(4))
