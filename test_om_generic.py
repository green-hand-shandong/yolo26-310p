"""通用 YOLO OM 真机测试：支持 e2e (1,300,6) 和原始头 (1,84,8400) 两种输出
延迟基准 + bus.jpg 检出统计（8400 输出在 host 侧做 argmax+conf 过滤）
"""
import acl
import numpy as np
import time
import sys

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

OM_PATH = sys.argv[1]
OUT_NPY = sys.argv[2]
N_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 100
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
check(acl.mdl.get_desc(model_id, model_desc), "get_desc")

in_size = acl.mdl.get_input_size_by_index(model_desc, 0)
out_size = acl.mdl.get_output_size_by_index(model_desc, 0)
n_float = out_size // 4

np_in = np.fromfile("/work/input_1x3x640x640.bin", dtype=np.float32)
dev_in, ret = acl.rt.malloc(in_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc in")
check(acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(np_in), in_size, ACL_MEMCPY_HOST_TO_DEVICE), "H2D")
dev_out, ret = acl.rt.malloc(out_size, ACL_MEM_MALLOC_HUGE_FIRST)
check(ret, "malloc out")

in_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_in, in_size)
check(acl.mdl.add_dataset_buffer(in_ds, r[0] if isinstance(r, (tuple, list)) else r), "add in")
out_ds = acl.mdl.create_dataset()
r = acl.create_data_buffer(dev_out, out_size)
check(acl.mdl.add_dataset_buffer(out_ds, r[0] if isinstance(r, (tuple, list)) else r), "add out")

for _ in range(10):
    check(acl.mdl.execute(model_id, in_ds, out_ds), "warmup")

host_out = np.empty(n_float, dtype=np.float32)
check(acl.rt.memcpy(acl.util.numpy_to_ptr(host_out), out_size, dev_out, out_size, ACL_MEMCPY_DEVICE_TO_HOST), "D2H")

# 输出格式判定
if n_float == 300 * 6:
    out = host_out.reshape(1, 300, 6)
    det = out[0][out[0][:, 4] > 0.25]
    fmt = 'e2e'
elif n_float == 84 * 8400:
    out = host_out.reshape(1, 84, 8400)
    o = out[0].T  # 8400,84
    scores = o[:, 4:]
    conf = scores.max(1)
    det = np.concatenate([o[conf > 0.25, :4], conf[conf > 0.25, None]], 1)
    fmt = 'raw84'
else:
    out = host_out.reshape(1, -1)
    det = []
    fmt = f'unknown{n_float}'

np.save(OUT_NPY, out)
print(f"model: {OM_PATH} fmt={fmt} dets(conf>0.25)={len(det)}")
if len(det):
    top = det[det[:, -1].argsort()[::-1][:3]]
    for r_ in top:
        print("   ", np.round(r_, 3))

lat = []
for _ in range(N_ITER):
    t0 = time.perf_counter()
    check(acl.mdl.execute(model_id, in_ds, out_ds), "execute")
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
print(f"latency ms: avg={lat.mean():.2f} p50={np.percentile(lat,50):.2f} p99={np.percentile(lat,99):.2f} min={lat.min():.2f}")
print(f"THROUGHPUT_IMG_S: {1 / (lat.mean() / 1000):.1f}")

check(acl.mdl.unload(model_id), "unload")
check(acl.rt.destroy_context(context), "ctx")
check(acl.rt.reset_device(DEV_ID), "reset")
check(acl.finalize(), "finalize")
print("FUNC TEST DONE")
