# 功能+数值验证: 加载预训练权重OM, 跑bus.jpg, 保存输出npy供host侧解码比对
import acl
import numpy as np
import time
import sys

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_DYNAMIC_TENSOR_NAME = "ascend_mbatch_shape_data"

OM_PATH = sys.argv[1] if len(sys.argv) > 1 else "/work/yolo26n_cann901_bs1.om"
IN_BIN = "/work/input_1x3x640x640.bin"
OUT_NPY = sys.argv[2] if len(sys.argv) > 2 else "/work/out_310p_bs1.npy"
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 1
IS_DYN = "dyn" in OM_PATH
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
print(f"model: {OM_PATH} inputs={n_in} outputs={acl.mdl.get_num_outputs(model_desc)}")

np_in = np.fromfile(IN_BIN, dtype=np.float32)
if BATCH > 1:
    np_in = np.tile(np_in, BATCH)
in_size = np_in.nbytes
out_size = acl.mdl.get_output_size_by_index(model_desc, 0)

dev_in, ret = acl.rt.malloc(in_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc in")
# 注意: memcpy(dst, dst_size, src, src_size, kind)
check(acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(np_in), in_size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")
dev_out, ret = acl.rt.malloc(out_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc out")

in_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_in, in_size)
buf_in = r[0] if isinstance(r, (tuple, list)) else r
check(acl.mdl.add_dataset_buffer(in_ds, buf_in), "add in buf")

dyn_idx = None
if IS_DYN:
    try:
        dyn_idx, ret = acl.mdl.get_input_index_by_name(model_desc, ACL_DYNAMIC_TENSOR_NAME)
        check(ret, "get dyn idx")
    except Exception:
        dyn_idx, ret = acl.mdl.get_input_index(model_desc, ACL_DYNAMIC_TENSOR_NAME)
        check(ret, "get dyn idx2")
    # 追加 mbatch_shape_data buffer
    mb_size = acl.mdl.get_input_size_by_index(model_desc, dyn_idx)
    dev_mb, ret = acl.rt.malloc(mb_size, ACL_MEM_MALLOC_HUGE_FIRST)
    check(ret, "malloc mb")
    r = acl.create_data_buffer(dev_mb, mb_size)
    buf_mb = r[0] if isinstance(r, (tuple, list)) else r
    check(acl.mdl.add_dataset_buffer(in_ds, buf_mb), "add mb buf")

out_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_out, out_size)
buf_out = r[0] if isinstance(r, (tuple, list)) else r
check(acl.mdl.add_dataset_buffer(out_ds, buf_out), "add out buf")

def set_batch(b):
    if not IS_DYN:
        return
    check(acl.mdl.set_dynamic_batch_size(model_id, model_desc, dyn_idx, b), f"set dyn batch {b}")
    mb = np.array([b], dtype=np.uint64)
    mb_size = acl.mdl.get_input_size_by_index(model_desc, dyn_idx)
    check(acl.rt.memcpy(dev_mb, mb_size, acl.util.numpy_to_ptr(mb), mb.nbytes, ACL_MEMCPY_HOST_TO_DEVICE), "mb H2D")

set_batch(BATCH)

# warmup
for _ in range(10):
    check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")

host_out = np.empty(out_size // 4, dtype=np.float32)
check(acl.rt.memcpy(acl.util.numpy_to_ptr(host_out), out_size, dev_out, out_size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H")
# 兼容不同任务头: 分类 (B,1000) / 姿态 (B,300,57) / 检测 (B,300,6)
if out_size // 4 == BATCH * 1000:
    out = host_out.reshape(BATCH, 1000)
elif out_size // 4 == BATCH * 300 * 57:
    out = host_out.reshape(BATCH, 300, 57)
else:
    out = host_out.reshape(BATCH, 300, 6)
np.save(OUT_NPY, out)
print(f"output shape: {out.shape}, saved to {OUT_NPY}")

# latency
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
lat = []
for _ in range(N):
    t0 = time.perf_counter()
    check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
print(f"latency ms: avg={lat.mean():.2f} p50={np.percentile(lat,50):.2f} p99={np.percentile(lat,99):.2f} min={lat.min():.2f}")
print(f"THROUGHPUT_IMG_S: {BATCH / (lat.mean() / 1000):.1f}")

check(acl.rt.free(dev_in), "free")
check(acl.rt.free(dev_out), "free")
check(acl.mdl.unload(model_id), "unload")
check(acl.rt.destroy_context(context), "ctx")
check(acl.rt.reset_device(DEV_ID), "reset")
check(acl.finalize(), "finalize")
print("FUNC TEST DONE")
