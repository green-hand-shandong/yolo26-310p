import acl
import numpy as np
OM = "/work/yolo26n_cls_310p.om"
IN_BIN = "/work/input_cls_1x3x224x224.bin"
ret = acl.init()
acl.rt.set_device(0)
ctx, ret = acl.rt.create_context(0)
mid, ret = acl.mdl.load_from_file(OM)
desc = acl.mdl.create_desc()
acl.mdl.get_desc(desc, mid)
np_in = np.fromfile("/work/input_cls_1x3x224x224.bin", dtype=np.float32)
in_size = np_in.nbytes
out_size = acl.mdl.get_output_size_by_index(desc, 0)
dev_in, ret = acl.rt.malloc(in_size, 0)
acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(np_in), in_size, 1)
dev_out, ret = acl.rt.malloc(out_size, 0)
ds_in = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_in, in_size)
acl.mdl.add_dataset_buffer(ds_in, r)
ds_out = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_out, out_size)
acl.mdl.add_dataset_buffer(ds_out, r)
for _ in range(5): acl.mdl.execute(mid, ds_in, ds_out)
host = np.empty(out_size // 4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_size, dev_out, out_size, 2)
print("om top-3:", host.argsort()[-3:][::-1], "max=", host.max())
