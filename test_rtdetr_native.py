import acl, numpy as np, sys, time, threading

OM = '/work/rtdetr_l_orig_310p_linux_aarch64.om'
def check(r, m):
    if isinstance(r, (tuple, list)): r = r[-1]
    if r != 0: print(f'[FAIL] {m} ret={r}', flush=True); sys.exit(1)

r = acl.init()
if r != 0 and r != 500000: check(r, 'init')
check(acl.rt.set_device(0), 'set_device')
ctx, r = acl.rt.create_context(0); check(r, 'ctx')
mid, r = acl.mdl.load_from_file(OM); check(r, 'load')
print('loaded', flush=True)
desc = acl.mdl.create_desc(); acl.mdl.get_desc(mid, desc)
in_size = acl.mdl.get_input_size_by_index(desc, 0)
out_size = acl.mdl.get_output_size_by_index(desc, 0)
print(f'in={in_size} out={out_size}', flush=True)
x = np.fromfile('/work/input_1x3x640x640.bin', dtype=np.float32).reshape(1,3,640,640)
dev_in, r = acl.rt.malloc(in_size, 0)
acl.rt.memcpy(dev_in, in_size, acl.util.numpy_to_ptr(x), in_size, 1)
dev_out, r = acl.rt.malloc(out_size, 0)
in_ds = acl.mdl.create_dataset()
b, r = acl.create_data_buffer(dev_in, in_size); acl.mdl.add_dataset_buffer(in_ds, b)
out_ds = acl.mdl.create_dataset()
b, r = acl.create_data_buffer(dev_out, out_size); acl.mdl.add_dataset_buffer(out_ds, b)
print('executing (timeout 120s)...', flush=True)
result = {}
def run():
    t0 = time.time()
    r = acl.mdl.execute(mid, in_ds, out_ds)
    result['ret'] = r; result['t'] = time.time()-t0
th = threading.Thread(target=run, daemon=True)
th.start(); th.join(120)
if th.is_alive():
    print('TIMEOUT: execute still running after 120s - device-side hang confirmed', flush=True)
    sys.exit(2)
print(f'execute ret={result["ret"]} time={result["t"]:.2f}s', flush=True)
host = np.empty(out_size//4, dtype=np.float32)
acl.rt.memcpy(acl.util.numpy_to_ptr(host), out_size, dev_out, out_size, 2)
o = host.reshape(1, 300, 6)[0]
d = o[o[:,4]>0.5]
print(f'dets(conf>0.5): {len(d)}', flush=True)
for r_ in d[:5]: print('  ', np.round(r_,3), flush=True)
