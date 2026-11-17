import acl, numpy as np, sys

def log(*a): print(*a, flush=True)

acl.init(); acl.rt.set_device(0); acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file('/tmp/mini_dd.om') if len(sys.argv)<2 else acl.mdl.load_from_file(sys.argv[1])
log('load:', r)
desc = acl.mdl.create_desc(); acl.mdl.get_desc(mid, desc)
in_size = acl.mdl.get_input_size_by_index(desc, 0)
out_size = acl.mdl.get_output_size_by_index(desc, 0)
# 构造已知输入: 0~1 线性
x = np.linspace(0, 1, 3*8*8, dtype=np.float32).reshape(1,3,8,8)
dev_in, r = acl.rt.malloc(in_size, 0)
acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(x), in_size, 1)
dev_out, r = acl.rt.malloc(out_size, 0)
in_ds = acl.mdl.create_dataset()
b = acl.create_data_buffer(dev_in, in_size); acl.mdl.add_dataset_buffer(in_ds, b)
out_ds = acl.mdl.create_dataset()
b = acl.create_data_buffer(dev_out, out_size); acl.mdl.add_dataset_buffer(out_ds, b)
r = acl.mdl.execute(mid, in_ds, out_ds)
log('execute:', r)
host = np.empty(out_size//4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_size, dev_out, out_size, 2)
np.save('/work/mini_npu_out.npy', host.reshape(1,1,8,8))
log('NPU out saved, stats: min', host.min(), 'max', host.max())

