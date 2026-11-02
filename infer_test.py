# pyACL smoke test: run yolo26n e2e OM on Ascend310P3
import acl
import numpy as np
import time
import sys

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

OM_PATH = "/work/yolo26n_e2e_310p.om"
IN_BIN = "/work/input_1x3x640x640.bin"
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
stream, ret = acl.rt.create_stream()
check(ret, "create_stream")

model_id, ret = acl.mdl.load_from_file(OM_PATH)
check(ret, "load om")
model_desc = acl.mdl.create_desc()
check(acl.mdl.get_desc(model_desc, model_id), "get_desc")

n_in = acl.mdl.get_num_inputs(model_desc)
n_out = acl.mdl.get_num_outputs(model_desc)
print(f"model inputs={n_in} outputs={n_out}")
for i in range(n_in):
    print(f"  in[{i}] size={acl.mdl.get_input_size_by_index(model_desc, i)}")
for i in range(n_out):
    print(f"  out[{i}] size={acl.mdl.get_output_size_by_index(model_desc, i)}")

np_in = np.fromfile(IN_BIN, dtype=np.float32)
in_size = acl.mdl.get_input_size_by_index(model_desc, 0)
out_size = acl.mdl.get_output_size_by_index(model_desc, 0)
assert np_in.nbytes == in_size, (np_in.nbytes, in_size)

dev_in, ret = acl.rt.malloc(in_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc in")
host_ptr = acl.util.numpy_to_ptr(np_in)
check(acl.rt.memcpy(dev_in, in_size, host_ptr, in_size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")

dev_out, ret = acl.rt.malloc(out_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc out")

in_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_in, in_size)
buf_in = r[0] if isinstance(r, (tuple, list)) else r
check(acl.mdl.add_dataset_buffer(in_ds, buf_in), "add buffer in")
out_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_out, out_size)
buf_out = r[0] if isinstance(r, (tuple, list)) else r
check(acl.mdl.add_dataset_buffer(out_ds, buf_out), "add buffer out")

# warmup
for _ in range(10):
    check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")

host_out = np.empty(out_size // 4, dtype=np.float32)
out_ptr = acl.util.numpy_to_ptr(host_out)
check(acl.rt.memcpy(out_ptr, out_size, dev_out, out_size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H")

out = host_out.reshape(1, 300, 6)
print("output shape:", out.shape)
print("score range: min=%.4f max=%.4f mean=%.4f" % (out[..., 4].min(), out[..., 4].max(), out[..., 4].mean()))
print("first 3 rows:\n", out[0, :3])

# latency
N = 100
t0 = time.perf_counter()
for _ in range(N):
    acl.mdl.execute(model_id, in_ds, out_ds)
dt = (time.perf_counter() - t0) / N * 1000
print(f"latency: {dt:.2f} ms/iter (bs=1, {N} iters, device sync incl.)")

check(acl.rt.free(dev_in), "free in")
check(acl.rt.free(dev_out), "free out")
check(acl.mdl.unload(model_id), "unload")
check(acl.rt.destroy_context(context), "destroy ctx")
check(acl.rt.reset_device(DEV_ID), "reset")
check(acl.finalize(), "finalize")
print("SMOKE TEST OK")
