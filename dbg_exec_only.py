import acl, numpy as np, sys, time, threading
OM = '/work/rtdetr_l_orig_310p_linux_aarch64.om'
def log(s): print(s, flush=True)
r = acl.init()
if r != 0 and r != 500000: log(f'init {r}'); sys.exit(1)
check = acl.rt.set_device(0); log(f'dev {check}')
ctx, r = acl.rt.create_context(0); log(f'ctx {r}')
acl.rt.set_context(ctx); log('set_context done')
mid, r = acl.mdl.load_from_file(OM); log(f'load {r}')
# 跳过 desc！直接按已知形状构造 dataset: input (1,3,640,640) f32, output (1,300,6) f32
in_size = 1*3*640*640*4
out_size = 1*300*6*4
log(f'assumed in={in_size} out={out_size}')
x = np.fromfile('/work/input_1x3x640x640.bin', dtype=np.float32)
dev_in, r = acl.rt.malloc(in_size, 0)
acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(x), in_size, 1)
dev_out, r = acl.rt.malloc(out_size, 0)
in_ds = acl.mdl.create_dataset()
b = acl.create_data_buffer(dev_in, in_size); acl.mdl.add_dataset_buffer(in_ds, b)
out_ds = acl.mdl.create_dataset()
b = acl.create_data_buffer(dev_out, out_size); acl.mdl.add_dataset_buffer(out_ds, b)
log('executing with 120s guard...')
res = {}
def run():
    t0 = time.time()
    r = acl.mdl.execute(mid, in_ds, out_ds)
    res['r'] = r; res['t'] = time.time()-t0
th = threading.Thread(target=run, daemon=True); th.start(); th.join(120)
if th.is_alive():
    log('HANG: execute not returned in 120s'); sys.exit(2)
log(f'execute ret={res["r"]} time={res["t"]:.2f}s')
host = np.empty(300*6, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_size, dev_out, out_size, 2)
o = host.reshape(300,6)
d = o[o[:,4]>0.5]
log(f'dets(conf>0.5): {len(d)}')
for r_ in d[:5]: log('  '+str(np.round(r_,3)))
np.save('/work/out_rtdetr_native.npy', o[None])
