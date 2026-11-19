import acl, numpy as np, sys
def log(s): print(s, flush=True)
def check(r, m):
    if isinstance(r, (tuple, list)): r = r[-1]
    if r != 0: log(f'[FAIL] {m} ret={r}'); sys.exit(1)
r = acl.init()
if r != 0 and r != 500000: log(f'init {r}'); sys.exit(1)
r = acl.rt.set_device(0); log(f'dev {r}')
ctx, r = acl.rt.create_context(0); check(r, 'ctx')
check(acl.rt.set_context(ctx), 'set_context')
mid, r = acl.mdl.load_from_file('/work/yolo26n_obb_92_310p_linux_aarch64.om'); check(r, 'load')
# C++ om_desc 方式确认输出个数——先用 pyACL 试 desc，段错误则跳过
try:
    desc = acl.mdl.create_desc()
    r = acl.mdl.get_desc(mid, desc)
    log(f'desc ret={r}')
    n_out = acl.mdl.get_num_outputs(desc)
    out_sizes = [acl.mdl.get_output_size_by_index(desc, i) for i in range(n_out)]
except Exception as e:
    log(f'desc failed: {e}')
    n_out = 1
    out_sizes = [300*7*4]
log(f'outputs={n_out} sizes={out_sizes}')
x = np.fromfile('/work/input_obb_1024.bin', dtype=np.float32)
in_size = 1*3*1024*1024*4
dev_in, r = acl.rt.malloc(in_size, 0)
acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(x), in_size, 1)
in_ds = acl.mdl.create_dataset()
b = acl.create_data_buffer(dev_in, in_size); acl.mdl.add_dataset_buffer(in_ds, b)
out_ds = acl.mdl.create_dataset()
dev_outs = []
for s in out_sizes:
    d, r = acl.rt.malloc(s, 0)
    dev_outs.append(d)
    b = acl.create_data_buffer(d, s); acl.mdl.add_dataset_buffer(out_ds, b)
log('executing...')
r = acl.mdl.execute(mid, in_ds, out_ds)
log(f'execute ret={r}')
host = np.empty(out_sizes[0]//4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_sizes[0], dev_outs[0], out_sizes[0], 2)
o = host.reshape(300, 7)
d = o[o[:,4]>0.25]
log(f'NPU OBB dets(conf>0.25): {len(d)}')
for r_ in d[:5]: log('  '+str(np.round(r_,3)))
np.save('/work/out_obb92.npy', o[None])
# CPU 对照
import onnxruntime as ort
s = ort.InferenceSession('/work/yolo26n-obb.onnx', providers=['CPUExecutionProvider'])
oc = s.run(None, {'images': x})[0][0]
dc = oc[oc[:,4]>0.25]
log(f'CPU OBB dets(conf>0.25): {len(dc)}')
if len(d) and len(dc):
    # 贪心匹配
    m = 0
    for r_ in dc:
        dist = np.abs(d[:,:4]-r_[:4]).max(axis=1)
        j = dist.argmin()
        if dist[j] < 0.05 and abs(d[j,4]-r_[4]) < 0.05 and d[j,6]==r_[6]: m += 1
    log(f'matched: {m}/{len(dc)}')
