#!/usr/bin/env python3
import acl
import numpy as np

OM = "/work/yolo26n_obb_310p.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc(); acl.mdl.get_desc(d, mid)
print("num in/out:", acl.mdl.get_num_inputs(d), acl.mdl.get_num_outputs(d))
for i in range(acl.mdl.get_num_inputs(d)):
    print("in", i, "dims:", acl.mdl.get_input_dims(d, i))
for i in range(acl.mdl.get_num_outputs(d)):
    print("out", i, "dims:", acl.mdl.get_output_dims(d, i))
np_in = np.fromfile("/work/input_obb_1x3x1024x1024.bin", dtype=np.float32)
in_size = np_in.nbytes
out_size = acl.mdl.get_output_size_by_index(d, 0)
dev_in, r = acl.rt.malloc(in_size, 0)
acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(np_in), in_size, 1)
dev_out, r = acl.rt.malloc(out_size, 0)
ds = acl.mdl.create_dataset()
r1 = acl.create_data_buffer(dev_in, in_size); acl.mdl.add_dataset_buffer(ds, r1)
r2 = acl.create_data_buffer(dev_out, out_size); acl.mdl.add_dataset_buffer(ds, r2)
ret = acl.mdl.execute(mid, ds, ds)
print("execute ret:", ret)
host = np.empty(out_size // 4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_size, dev_out, out_size, 2)
print("out stats: min=%.4f max=%.4f mean=%.4f" % (host.min(), host.max(), host.mean()))
