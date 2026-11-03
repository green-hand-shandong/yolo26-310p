#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yolo26n-obb 310P 真机验证: 按 desc 枚举全部输入/输出 buffer, 推理后保存全部输出 npy"""
import acl
import numpy as np
import sys

ACL_MEM_MALLOC_HUGE_FIRST = 0

OM_PATH = sys.argv[1]
IN_NPY = sys.argv[2]
OUT_PREFIX = sys.argv[3]
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
n_in = acl.mdl.get_num_inputs(desc)
n_out = acl.mdl.get_num_outputs(desc)
print(f"model: inputs={n_in} outputs={n_out}")

# 枚举全部输入, 主输入从文件读, 其余(常量)置零
np_in = np.fromfile(IN_NPY, dtype=np.float32)
dev_ins, ds = [], acl.mdl.create_dataset()
for i in range(n_in):
    sz = acl.mdl.get_input_size_by_index(desc, i)
    if i == 0:
        data = np_in
        assert data.nbytes == sz, (data.nbytes, sz)
    else:
        data = np.zeros(sz // 4, dtype=np.float32)
        print(f"  const input[{i}] size={sz} zero-filled")
    dev_i, ret = acl.rt.malloc(sz, ACL_MEM_MALLOC_HUGE_FIRST)
    check(ret, f"malloc in{i}")
    check(acl.rt.memcpy(dev_i, sz, acl.util.numpy_to_ptr(data), sz, 1), f"H2D in{i}")
    r = acl.create_data_buffer(dev_i, sz)
    buf = r[0] if isinstance(r, (tuple, list)) else r
    check(acl.mdl.add_dataset_buffer(ds, buf), f"add in{i} buf")
    dev_ins.append(dev_i)

# 枚举全部输出
dev_outs, out_sizes = [], []
for i in range(n_out):
    sz = acl.mdl.get_output_size_by_index(desc, i)
    dev_o, ret = acl.rt.malloc(sz, ACL_MEM_MALLOC_HUGE_FIRST)
    check(ret, f"malloc out{i}")
    r = acl.create_data_buffer(dev_o, sz)
    buf = r[0] if isinstance(r, (tuple, list)) else r
    check(acl.mdl.add_dataset_buffer(ds, buf), f"add out{i} buf")
    dev_outs.append(dev_o)
    out_sizes.append(sz)

for _ in range(10):
    check(acl.mdl.execute(model_id, ds, ds), "execute")

for i, sz in enumerate(out_sizes):
    host = np.empty(sz // 4, dtype=np.float32)
    check(acl.rt.memcpy(acl.util.numpy_to_ptr(host), sz, dev_outs[i], sz, 2), f"D2H out{i}")
    np.save(f"{OUT_PREFIX}_{i}.npy", host)
    print(f"out[{i}] size={sz} saved {OUT_PREFIX}_{i}.npy")

for d in dev_ins + dev_outs:
    acl.rt.free(d)
check(acl.mdl.unload(model_id), "unload")
check(acl.rt.destroy_context(context), "ctx")
check(acl.rt.reset_device(DEV_ID), "reset")
check(acl.finalize(), "finalize")
print("OBB FUNC TEST DONE")
