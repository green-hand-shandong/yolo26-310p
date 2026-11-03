#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yolo26n-pose 310P 真机验证: 加载 OM, 跑测试图, 保存输出 npy 供 host 侧解码比对 + 测速"""
import acl
import numpy as np
import time
import sys

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

OM_PATH = sys.argv[1] if len(sys.argv) > 1 else "/work/yolo26n_pose_310p.om"
IN_BIN = sys.argv[2] if len(sys.argv) > 2 else "/work/input_pose_1x3x640x640.bin"
OUT_NPY = sys.argv[3] if len(sys.argv) > 3 else "/work/out_pose_310p.npy"
DEV_ID = 0

def check(ret, msg):
    if isinstance(ret, (tuple, list)):
        ret = ret[-1]
    if ret != 0:
        print(f"[FAIL] {msg} ret={ret}")
        sys.exit(1)

ret = acl.init()
if ret != 0 and ret != 500000:
    check(ret, "acl.init")
check(acl.rt.set_device(DEV_ID), "set_device")
context, ret = acl.rt.create_context(DEV_ID)
check(ret, "create_context")

model_id, ret = acl.mdl.load_from_file(OM_PATH)
check(ret, "load om")
model_desc = acl.mdl.create_desc()
check(acl.mdl.get_desc(model_desc, model_id), "get_desc")
print(f"model: {OM_PATH} inputs={acl.mdl.get_num_inputs(model_desc)} "
      f"outputs={acl.mdl.get_num_outputs(model_desc)}")

np_in = np.fromfile(IN_BIN, dtype=np.float32)
in_size = np_in.nbytes
out_size = acl.mdl.get_output_size_by_index(model_desc, 0)

dev_in, ret = acl.rt.malloc(in_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc in")
# memcpy(dst, dst_size, src, src_size, kind)
check(acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(np_in), in_size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")
dev_out, ret = acl.rt.malloc(out_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc out")

in_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_in, in_size)
buf_in = r[0] if isinstance(r, (tuple, list)) else r
check(acl.mdl.add_dataset_buffer(in_ds, buf_in), "add in buf")
out_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_out, out_size)
buf_out = r[0] if isinstance(r, (tuple, list)) else r
check(acl.mdl.add_dataset_buffer(out_ds, buf_out), "add out buf")

for _ in range(10):
    check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")

host_out = np.empty(out_size // 4, dtype=np.float32)
check(acl.rt.memcpy(acl.util.numpy_to_ptr(host_out), out_size, dev_out, out_size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H")
out = host_out.reshape(1, 300, 57)
np.save(OUT_NPY, out)
print(f"output shape: {out.shape}, saved to {OUT_NPY}")

N = 100
lat = []
for _ in range(N):
    t0 = time.perf_counter()
    check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
print(f"latency ms: avg={lat.mean():.2f} p50={np.percentile(lat,50):.2f} p99={np.percentile(lat,99):.2f}")
print("FUNC TEST DONE")

check(acl.rt.free(dev_in), "free in")
check(acl.rt.free(dev_out), "free out")
check(acl.mdl.unload(model_id), "unload")
check(acl.rt.destroy_context(context), "ctx")
check(acl.rt.reset_device(DEV_ID), "reset")
check(acl.finalize(), "finalize")
