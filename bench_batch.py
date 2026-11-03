import acl, numpy as np, time, sys

OM = sys.argv[1]; BS = int(sys.argv[2]); N_ITER = int(sys.argv[3]) if len(sys.argv)>3 else 50
acl.init(); acl.rt.set_device(0)
ctx,_ = acl.rt.create_context(0)
mid,_ = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc(); acl.mdl.get_desc(d, mid)
in_sz = acl.mdl.get_input_size_by_index(d, 0)
out_sz = acl.mdl.get_output_size_by_index(d, 0)
print(f"bs={BS}: in={in_sz}B out={out_sz}B")
base = np.fromfile('/work/input_1x3x640x640.bin', dtype=np.float32).reshape(1,3,640,640)
batch_in = np.tile(base, (BS,1,1,1)).copy()
dev_i,_ = acl.rt.malloc(in_sz, 0); dev_o,_ = acl.rt.malloc(out_sz, 0)
ds_i = acl.mdl.create_dataset(); r1=acl.create_data_buffer(dev_i,in_sz); acl.mdl.add_dataset_buffer(ds_i, r1[0] if isinstance(r1,(tuple,list)) else r1)
ds_o = acl.mdl.create_dataset(); r2=acl.create_data_buffer(dev_o,out_sz); acl.mdl.add_dataset_buffer(ds_o, r2[0] if isinstance(r2,(tuple,list)) else r2)
hbuf = np.empty(in_sz//4, dtype=np.float32); hout = np.empty(out_sz//4, dtype=np.float32)
for _ in range(5): acl.mdl.execute(mid, ds_i, ds_o)
t0=time.perf_counter()
for _ in range(N_ITER):
    hbuf[:] = batch_in.ravel()
    acl.rt.memcpy(dev_i, in_sz, acl.util.numpy_to_ptr(hbuf), in_sz, 1)
    acl.mdl.execute(mid, ds_i, ds_o)
    acl.rt.memcpy(acl.util.numpy_to_ptr(hout), out_sz, dev_o, out_sz, 2)
dt = time.perf_counter()-t0
print(f"bs={BS}: {dt*1000/N_ITER:.2f} ms/batch = {dt*1000/N_ITER/BS:.2f} ms/img = {BS*N_ITER/dt:.1f} img/s")
