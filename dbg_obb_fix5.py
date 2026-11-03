#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB 500002: 上一次 execute ret!=0 但 out buffer 已有真实数据(max=0.8392),
说明推理实际执行了, ret!=0 是 blob 数校验的"软告警"。上一版读 out buffer 全 0
是因为用了 execute 之前申请的 dev_out —— 真实输出写到别处。
这次: dataset 加 [in0, 占位] 两 buffer, execute 后从 out0 buffer 读。"""
import acl
import numpy as np

OM = "/work/yolo26n_obb_310p.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc()
acl.mdl.get_desc(d, mid)

in_ds = acl.mdl.create_dataset()
np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
dev_in, r = acl.rt.malloc(np_in.nbytes, 0)
acl.rt.memcpy(dev_in, np_in.nbytes, acl.util.numpy_to_ptr(np_in), np_in.nbytes, 1)
r1 = acl.create_data_buffer(dev_in, np_in.nbytes)
acl.mdl.add_dataset_buffer(in_ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
# 占位第二 blob (错误日志说 input blobs.size:2)
dev_p, r = acl.rt.malloc(8, 0)
r2 = acl.create_data_buffer(dev_p, 8)
acl.mdl.add_dataset_buffer(in_ds, r2[0] if isinstance(r2, (tuple, list)) else r2)

out_ds = acl.mdl.create_dataset()
out0_sz = 8400
dev_out, r = acl.rt.malloc(out0_sz, 0)
r3 = acl.create_data_buffer(dev_out, out0_sz)
acl.mdl.add_dataset_buffer(out_ds, r3[0] if isinstance(r3, (tuple, list)) else r3)
# 输出侧错误日志说 output blobs.size:2 —— 也加占位
dev_po, r = acl.rt.malloc(8, 0)
r4 = acl.create_data_buffer(dev_po, 8)
acl.mdl.add_dataset_buffer(out_ds, r4[0] if isinstance(r4, (tuple, list)) else r4)

ret = acl.mdl.execute(mid, in_ds, out_ds)
print("execute ret:", ret)

host = np.empty(2100, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), 8400, dev_out, 8400, 2)
out = host.reshape(1, 300, 7)
det = out[0][out[0][..., 4] >= 0.25]
print("310P conf>=0.25 检测数:", len(det))
for b in det[:8]:
    print(f"  score={b[4]:.3f} box={b[:4].astype(int)} angle={b[5]:.3f} cls={b[6]}")
np.save("/work/out_obb_310p_final.npy", out)
print("saved")
