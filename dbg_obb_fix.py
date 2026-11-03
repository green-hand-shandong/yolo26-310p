#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB execute 500002 修复尝试: dataset 里额外添加 index=1..2 的隐藏 blob (size=0)"""
import acl
import numpy as np

OM = "/work/yolo26n_obb_310p.om"
IN_BIN = "/work/input_obb_1x3x1024x1024.bin"
OUT_PREFIX = "/work/out_obb310p_fix"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc()
acl.mdl.get_desc(d, mid)

ds = acl.mdl.create_dataset()
np_in = np.fromfile(IN_BIN, dtype=np.float32)
in0_sz = np_in.nbytes
dev0, r = acl.rt.malloc(in0_sz, 0)
acl.rt.memcpy(dev0, in0_sz, acl.util.numpy_to_ptr(np_in), in0_sz, 1)
r1 = acl.create_data_buffer(dev0, in0_sz)
buf0 = r1[0] if isinstance(r1, (tuple, list)) else r1
print("add in[0]:", acl.mdl.add_dataset_buffer(ds, buf0))

# 添加隐藏的 in[1] (size=0)
try:
    sz1 = acl.mdl.get_input_size_by_index(d, 1)
    print("in[1] size:", sz1)
    dev1, r = acl.rt.malloc(sz1 if sz1 > 0 else 8, 0)
    r2 = acl.create_data_buffer(dev1, sz1)
    buf1 = r2[0] if isinstance(r2, (tuple, list)) else r2
    print("add in[1]:", acl.mdl.add_dataset_buffer(ds, buf1))
except Exception as e:
    print("in[1] add err:", str(e)[:120])

# 输出侧 out[0] + 隐藏 out[1]
out0_sz = acl.mdl.get_output_size_by_index(d, 0)
dev_o0, r = acl.rt.malloc(out0_sz, 0)
r3 = acl.create_data_buffer(dev_o0, out0_sz)
bufo0 = r3[0] if isinstance(r3, (tuple, list)) else r3
print("add out[0]:", acl.mdl.add_dataset_buffer(ds, bufo0))
try:
    szo1 = acl.mdl.get_output_size_by_index(d, 1)
    print("out[1] size:", szo1)
    dev_o1, r = acl.rt.malloc(szo1 if szo1 > 0 else 8, 0)
    r4 = acl.create_data_buffer(dev_o1, szo1)
    bufo1 = r4[0] if isinstance(r4, (tuple, list)) else r4
    print("add out[1]:", acl.mdl.add_dataset_buffer(ds, bufo1))
except Exception as e:
    print("out[1] add err:", str(e)[:120])

ret = acl.mdl.execute(mid, ds, ds)
print("execute ret:", ret)
if ret == 0 or (isinstance(ret, (tuple, list)) and ret[-1] == 0):
    host = np.empty(out0_sz // 4, dtype=np.float32)
    acl.rt.memcpy(acl.util.numpy_to_ptr(host), out0_sz, dev_o0, out0_sz, 2)
    print("out stats: min=%.4f max=%.4f" % (host.min(), host.max()))
    np.save(OUT_PREFIX + "_0.npy", host.reshape(1, 300, 7))
    print("saved:", OUT_PREFIX + "_0.npy")
