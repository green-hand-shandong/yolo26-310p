import acl, numpy as np, sys, time, threading
OM = '/work/rtdetr_l_92_310p.om'
def log(s): print(s, flush=True)
def check(r, m):
    if isinstance(r, (tuple, list)): r = r[-1]
    if r != 0: log(f'[FAIL] {m} ret={r}'); sys.exit(1)
r = acl.init()
if r != 0 and r != 500000: check(r, 'init')
check(acl.rt.set_device(0), 'set_device')
ctx, r = acl.rt.create_context(0); check(r, 'ctx')
check(acl.rt.set_context(ctx), 'set_context')
mid, r = acl.mdl.load_from_file(OM); check(r, 'load')
# desc 查询对含 Ascend C 算子的 OM 会段错误（pyACL），用 C++ om_desc 已确认:
# inputs=1 (4915200), outputs=5 [7200, 4800, 96000, 4800, 96000]
n_in, n_out = 1, 5
in_sizes = [4915200]
out_sizes = [7200, 4800, 96000, 4800, 96000]
log(f'inputs={n_in} outputs={n_out}')
log(f'out_sizes={out_sizes}')
x = np.fromfile('/work/input_1x3x640x640.bin', dtype=np.float32)
dev_ins = []
in_ds = acl.mdl.create_dataset()
for s in in_sizes:
    d, r = acl.rt.malloc(s, 0); check(r, 'malloc in')
    dev_ins.append(d)
    b = acl.create_data_buffer(d, s); check(acl.mdl.add_dataset_buffer(in_ds, b), 'add in')
check(acl.rt.memcpy(dev_ins[0], in_sizes[0], acl.util.numpy_to_ptr(x), in_sizes[0], 1), 'H2D')
out_ds = acl.mdl.create_dataset()
dev_outs = []
for s in out_sizes:
    d, r = acl.rt.malloc(s, 0); check(r, 'malloc out')
    dev_outs.append(d)
    b = acl.create_data_buffer(d, s); check(acl.mdl.add_dataset_buffer(out_ds, b), 'add out')
check(acl.rt.set_context(ctx), 're-set_context')
log('executing...')
res = {}
def run():
    t0 = time.time()
    r = acl.mdl.execute(mid, in_ds, out_ds)
    res['r'] = r; res['t'] = time.time()-t0
th = threading.Thread(target=run, daemon=True); th.start(); th.join(200)
if th.is_alive():
    log('HANG after 200s'); sys.exit(2)
log(f'execute ret={res["r"]} time={res["t"]:.2f}s')
# out[0] 应为 (300,6) 主输出
host = np.empty(out_sizes[0]//4, dtype=np.float32)
check(acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_sizes[0], dev_outs[0], out_sizes[0], 2), 'D2H')
o = host.reshape(300, 6)
d = o[o[:,4]>0.5]
log(f'dets(conf>0.5): {len(d)}')
for r_ in d[:5]: log('  '+str(np.round(r_,3)))
np.save('/work/out_rtdetr_92.npy', o[None])
log('SAVED')
