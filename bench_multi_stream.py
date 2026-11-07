"""host 提交瓶颈验证：单 stream vs 多 stream 并发推理吞吐（同芯片）
原理：若 host 提交是瓶颈，多 stream 异步流水应显著提升；若 device 是瓶颈则 ~1x
"""
import acl, numpy as np, time, sys

N_STREAMS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
OM = '/data2/gxw/yolo26-310p-eval/yolo26n_cann901_bs1.om'

acl.init()
acl.rt.set_device(0)
ctx, _ = acl.rt.create_context(0)

model_id, ret = acl.mdl.load_from_file(OM)
md = acl.mdl.create_desc(); acl.mdl.get_desc(md, model_id)
in_sz = acl.mdl.get_input_size_by_index(md, 0)
out_sz = acl.mdl.get_output_size_by_index(md, 0)

np_in = np.fromfile('/data2/gxw/yolo26-310p-eval/input_1x3x640x640.bin', dtype=np.float32)

# 每 stream 独立 dataset + stream
streams, ds_is, ds_os = [], [], []
for _ in range(N_STREAMS):
    dev_in, _ = acl.rt.malloc(in_sz, 0)
    acl.rt.memcpy(dev_in, in_sz, acl.util.numpy_to_ptr(np_in), in_sz, 1)
    dev_out, _ = acl.rt.malloc(out_sz, 0)
    ds_i = acl.mdl.create_dataset()
    b = acl.create_data_buffer(dev_in, in_sz); b = b[0] if isinstance(b, (tuple, list)) else b
    acl.mdl.add_dataset_buffer(ds_i, b)
    ds_o = acl.mdl.create_dataset()
    b = acl.create_data_buffer(dev_out, out_sz); b = b[0] if isinstance(b, (tuple, list)) else b
    acl.mdl.add_dataset_buffer(ds_o, b)
    s, _ = acl.rt.create_stream()
    streams.append(s); ds_is.append(ds_i); ds_os.append(ds_o)

# warmup
for s in streams:
    for _ in range(10):
        acl.mdl.execute_async(model_id, ds_is[0], ds_os[0], s)
for s in streams:
    acl.rt.synchronize_stream(s)

# 流水提交：round-robin 多 stream，末尾统一同步
t0 = time.perf_counter()
for i in range(N_ITER):
    acl.mdl.execute_async(model_id, ds_is[i % N_STREAMS], ds_os[i % N_STREAMS], streams[i % N_STREAMS])
for s in streams:
    acl.rt.synchronize_stream(s)
total = time.perf_counter() - t0
print(f"streams={N_STREAMS}: {N_ITER} iters in {total:.2f}s = {N_ITER/total:.0f} fps ({total/N_ITER*1000:.2f} ms/img)")
