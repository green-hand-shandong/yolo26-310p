#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB execute 500002 诊断2: dataset blob 数量不匹配。查看 blobs.size:2 的实际 blob 数期望。
绕过方案: 模型在 device 端真正期望的 blob 数 —— 用 pyACL 试 '2 组 dataset buffer' 组合,
即 dataset 包含 [主输入, 隐藏 blob(size=8 占位)], 输出同理。"""
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
# 输入: 主输入
np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
dev_in, r = acl.rt.malloc(np_in.nbytes, 0)
acl.rt.memcpy(dev_in, np_in.nbytes, acl.util.numpy_to_ptr(np_in), np_in.nbytes, 1)
r1 = acl.create_data_buffer(dev_in, np_in.nbytes)
acl.mdl.add_dataset_buffer(ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
# 第二个 size=0 占位 blob —— 用 size=8 buffer 模拟
dev_p, r = acl.rt.malloc(8, 0)
r2 = acl.create_data_buffer(dev_p, 8)
acl.mdl.add_dataset_buffer(ds, r2[0] if isinstance(r2, (tuple, list)) else r2)

ret = acl.mdl.execute(mid, ds, ds)
print("execute ret:", ret)

host = np.empty(2100, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), 8400, dev_in, 8400, 2)
print("out stats: min=%.4f max=%.4f" % (host.min(), host.max()))
