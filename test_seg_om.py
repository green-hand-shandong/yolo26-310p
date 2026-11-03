#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yolo26n-seg 310P 真机验证: 多输出模型 (检测 1,300,38 + proto 1,32,160,160), 全输出分配 buffer + 测速"""
import acl
import numpy as np
import time
import sys

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

OM_PATH = sys.argv[1] if len(sys.argv) > 1 else "/work/yolo26n_seg_310p.om"
IN_BIN = sys.argv[2] if len(sys.argv) > 2 else "/work/input_seg_1x3x640x640.bin"
OUT_PREFIX = sys.argv[3] if len(sys.argv) > 3 else "/work/out_seg_310p"
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

n_in = acl.mdl.get_num_inputs(model_desc)
n_out = acl.mdl.get_num_outputs(model_desc)
out_sizes = [acl.mdl.get_output_size_by_index(model_desc, i) for i in range(n_out)]
print(f"model: {OM_PATH} inputs={n_in} outputs={n_out} out_sizes={out_sizes}")

np_in = np.fromfile(IN_BIN, dtype=np.float32)
in_size = np_in.nbytes

dev_in, ret = acl.rt.malloc(in_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc in")
check(acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(np_in), in_size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")

in_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_in, in_size)
buf_in = r[0] if isinstance(r, (tuple, list)) else r
check(acl.mdl.add_dataset_buffer(in_ds, buf_in), "add in buf")

# 关键: 为每一个输出分配 buffer 并全部注册
dev_outs, out_dss = [], acl.mdl.create_dataset()
for i, sz in enumerate(out_sizes):
    dev_o, ret = acl.rt.malloc(sz, ACL_MEM_MALLOC_HUGE_FIRST)
    check(ret, f"malloc out{i}")
    r = acl.create_data_buffer(dev_o, sz)
    buf = r[0] if isinstance(r, (tuple, list)) else r
    check(acl.mdl.add_dataset_buffer(out_dss, buf), f"add out{i} buf")
    dev_outs.append(dev_o)

for _ in range(10):
    check(acl.mdl.execute(model_id, in_ds, out_dss), "execute")

# 全输出回传
for i, sz in enumerate(out_sizes):
    host = np.empty(sz // 4, dtype=np.float32)
    check(acl.rt.memcpy(acl.util.numpy_to_ptr(host), sz, dev_outs[i], sz, ACL_MEMCPY_DEVICE_TO_HOST), f"D2H out{i}")
    np.save(f"{OUT_PREFIX}_{i}.npy", host)
    print(f"out[{i}] shape={host.shape} saved {OUT_PREFIX}_{i}.npy")

N = 100
lat = []
for _ in range(N):
    t0 = time.perf_counter()
    check(acl.mdl.execute(model_id, in_ds, out_dss), "execute")
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
print(f"latency ms: avg={lat.mean():.2f} p50={np.percentile(lat,50):.2f} p99={np.percentile(lat,99):.2f}")
print("SEG FUNC TEST DONE")

check(acl.rt.free(dev_in), "free in")
for d in dev_outs:
    check(acl.rt.free(d), "free out")
check(acl.mdl.unload(model_id), "unload")
check(acl.rt.destroy_context(context), "ctx")
check(acl.rt.reset_device(DEV_ID), "reset")
check(acl.finalize(), "finalize")
