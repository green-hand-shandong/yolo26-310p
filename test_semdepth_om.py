#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sem/depth 310P 真机验证: bus.jpg 输入, 推理后保存输出 npy 并做基本 sanity"""
import acl
import numpy as np
import sys

ACL_MEM_MALLOC_HUGE_FIRST = 0

OM_PATH = sys.argv[1]
IN_BIN = sys.argv[2]
OUT_NPY = sys.argv[3]
DEV_ID = 0


def check(ret, msg):
    if isinstance(ret, (tuple, list)):
        ret = ret[-1]
    if ret != 0:
        print(f"[FAIL] {msg} ret={ret}")
        sys.exit(1)


ret = acl.init()
if ret not in (0, 500000):
    print("acl.init:", ret)
check(acl.rt.set_device(DEV_ID), "set_device")
context, ret = acl.rt.create_context(DEV_ID)
check(ret, "create_context")

model_id, ret = acl.mdl.load_from_file(OM_PATH)
check(ret, "load om")
desc = acl.mdl.create_desc()
check(acl.mdl.get_desc(desc, model_id), "get_desc")
n_in, n_out = acl.mdl.get_num_inputs(desc), acl.mdl.get_num_outputs(desc)
print(f"model: {OM_PATH} inputs={n_in} outputs={n_out}")
for i in range(n_out):
    print("  out dims:", acl.mdl.get_output_dims(desc, i)[0])

np_in = np.fromfile(IN_BIN, dtype=np.float32)
in_ds = acl.mdl.create_dataset()
dev_in, r = acl.rt.malloc(np_in.nbytes, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc in")
check(acl.rt.memcpy(dev_in, np_in.nbytes, acl.util.numpy_to_ptr(np_in), np_in.nbytes, 1), "H2D")
r1 = acl.create_data_buffer(dev_in, np_in.nbytes)
check(acl.mdl.add_dataset_buffer(in_ds, r1[0] if isinstance(r1, (tuple, list)) else r1), "add in buf")

out_ds = acl.mdl.create_dataset()
dev_outs, out_sizes = [], []
for i in range(n_out):
    sz = acl.mdl.get_output_size_by_index(desc, i)
    dev_o, r = acl.rt.malloc(sz, ACL_MEM_MALLOC_HUGE_FIRST)
    check(r, f"malloc out{i}")
    r2 = acl.create_data_buffer(dev_o, sz)
    check(acl.mdl.add_dataset_buffer(out_ds, r2[0] if isinstance(r2, (tuple, list)) else r2), f"add out{i} buf")
    dev_outs.append(dev_o)
    out_sizes.append(sz)

import time
for _ in range(10):
    check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")

N = 100
lat = []
for _ in range(N):
    t0 = time.perf_counter()
    check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)

for i, sz in enumerate(out_sizes):
    host = np.empty(sz // 4, dtype=np.float32)
    check(acl.rt.memcpy(acl.util.numpy_to_ptr(host), sz, dev_outs[i], sz, 2), f"D2H out{i}")
    np.save(OUT_NPY, host)
    print(f"out[{i}] shape={host.shape} min={host.min():.4f} max={host.max():.4f} mean={host.mean():.4f} saved {OUT_NPY}")

print(f"latency ms: avg={lat.mean():.2f} p50={np.percentile(lat,50):.2f} p99={np.percentile(lat,99):.2f}")
print("SEM/DEPTH 310P TEST DONE")
