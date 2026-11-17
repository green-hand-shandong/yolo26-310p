import acl
import numpy as np
import sys

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

OM_PATH = sys.argv[1]
DEV_ID = 0

def check(ret, msg):
    if isinstance(ret, (tuple, list)):
        ret = ret[-1]
    if ret != 0:
        print(f"[FAIL] {msg} ret={ret}", flush=True)
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
in_size = acl.mdl.get_input_size_by_index(model_desc, 0)
out_size = acl.mdl.get_output_size_by_index(model_desc, 0)
print(f"load ok: in_size={in_size} out_size={out_size}", flush=True)

# 已知输入: 0~1 线性
H = W = 8
x = np.linspace(0, 1, 3*H*W, dtype=np.float32).reshape(1,3,H,W)
np.save("/work/mini_input.npy", x)

dev_in, ret = acl.rt.malloc(in_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc in")
check(acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(x), in_size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")
dev_out, ret = acl.rt.malloc(out_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc out")
in_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_in, in_size)
check(acl.mdl.add_dataset_buffer(in_ds, r[0] if isinstance(r, (tuple, list)) else r), "add in")
out_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_out, out_size)
check(acl.mdl.add_dataset_buffer(out_ds, r[0] if isinstance(r, (tuple, list)) else r), "add out")
check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")
host = np.empty(out_size // 4, dtype=np.float32)
check(acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_size, dev_out, out_size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H")
np.save("/work/mini_npu_out.npy", host.reshape(1, 1, H, W))
print(f"NPU out stats: min={host.min():.6f} max={host.max():.6f} mean={host.mean():.6f}", flush=True)
print("MINI TEST DONE")
