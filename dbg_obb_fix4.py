#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB execute 500002: 上次读的是 dev_in 错误 —— 输出 buffer 是 dev_o0。
正确做法: dataset 加占位 blob 触发执行, 从真实输出 buffer 读结果。"""
import acl
import numpy as np

OM = "/work/yolo26n_obb_310p.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc()
acl.mdl.get_desc(d, mid)

ds = acl.mdl.create_dataset()
np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
dev_in, r = acl.rt.malloc(np_in.nbytes, 0)
acl.rt.memcpy(dev_in, np_in.nbytes, acl.util.numpy_to_ptr(np_in), np_in.nbytes, 1)
r1 = acl.create_data_buffer(dev_in, np_in.nbytes)
acl.mdl.add_dataset_buffer(ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
dev_p, r = acl.rt.malloc(8, 0)
r2 = acl.create_data_buffer(dev_p, 8)
acl.mdl.add_dataset_buffer(ds, r2[0] if isinstance(r2, (tuple, list)) else r2)

out0_sz = acl.mdl.get_output_size_by_index(d, 0)
dev_out, r = acl.rt.malloc(out0_sz, 0)
r3 = acl.create_data_buffer(dev_out, out0_sz)
acl.mdl.add_dataset_buffer(ds, r3[0] if isinstance(r3, (tuple, list)) else r3)
# dataset: [in0, 占位, out0] —— 顺序不对; execute 的 dataset 不区分 in/out, 顺序按模型
# 改为: 单独 in_ds 与 out_ds
in_ds = acl.mdl.create_dataset()
acl.mdl.add_dataset_buffer(in_ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
acl.mdl.add_dataset_buffer(in_ds, r2[0] if isinstance(r2, (tuple, list)) else r2)
out_ds = acl.mdl.create_dataset()
acl.mdl.add_dataset_buffer(out_ds, r3[0] if isinstance(r3, (tuple, list)) else r3)

for _ in range(10):
    ret = acl.mdl.execute(mid, in_ds, out_ds)  # 忽略 ret, 直接读 buffer

host = np.empty(out0_sz // 4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), out0_sz, dev_out, out0_sz, 2)
out = host.reshape(1, 300, 7)
det = out[0][out[0][..., 4] >= 0.25]
print("execute ret:", ret)
print("310P conf>=0.25 检测数:", len(det))
for b in det[:6]:
    print(f"  score={b[4]:.3f} box={b[:4].astype(int)} angle={b[5]:.3f} cls={b[6]}")
np.save("/work/out_obb_310p_final.npy", out)
print("saved out_obb_310p_final.npy")
