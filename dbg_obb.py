#!/usr/bin/env python3
import acl
import numpy as np
import sys

OM = "/work/yolo26n_obb_310p.om"
ret = acl.init()
check = lambda r, m: print(f"{'OK' if (r[-1] if isinstance(r,(tuple,list)) else r)==0 else 'FAIL'} {m}")
check(acl.rt.set_device(0), "setdev")
ctx, r = acl.rt.create_context(0); print("ctx:", r)
mid, r = acl.mdl.load_from_file(OM); print("load:", r)
d = acl.mdl.create_desc(); acl.mdl.get_desc(d, mid)
n_in = acl.mdl.get_num_inputs(d)
n_out = acl.mdl.get_num_outputs(d)
print("num in/out:", n_in, n_out)
for i in range(n_in):
    print(" in", i, "size:", acl.mdl.get_input_size_by_index(d, i))
for i in range(n_out):
    print(" out", i, "size:", acl.mdl.get_output_size_by_index(d, i))
np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
in_size = np_in.nbytes
print("input file bytes:", in_size)
out_size = acl.mdl.get_output_size_by_index(d, 0)
dev_in, r = acl.rt.malloc(in_size, 0); print("malloc in:", r)
acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(np_in), in_size, 1)
dev_out, r = acl.rt.malloc(out_size, 0); print("malloc out:", r)
ds = acl.mdl.create_dataset()
r1 = acl.create_data_buffer(dev_in, in_size); acl.mdl.add_dataset_buffer(ds, r1)
r2 = acl.create_data_buffer(dev_out, out_size); acl.mdl.add_dataset_buffer(ds, r2)
ret = acl.mdl.execute(mid, ds, ds)
print("execute ret:", ret)
