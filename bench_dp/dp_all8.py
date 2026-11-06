"""8 芯片全开：DP 扩展极限"""
import sys, os, time
import multiprocessing as mp

def worker(global_chip, n, q):
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(global_chip)
    import acl, numpy as np
    acl.init()
    acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)
    model_id, _ = acl.mdl.load_from_file('/work/yolo26n_cann901_bs1.om')
    md = acl.mdl.create_desc(); acl.mdl.get_desc(md, model_id)
    in_sz = acl.mdl.get_input_size_by_index(md, 0)
    out_sz = acl.mdl.get_output_size_by_index(md, 0)
    np_in = np.fromfile('/work/input_1x3x640x640.bin', dtype=np.float32)
    dev_in, _ = acl.rt.malloc(in_sz, 0)
    acl.rt.memcpy(dev_in, in_sz, acl.util.numpy_to_ptr(np_in), in_sz, 1)
    dev_out, _ = acl.rt.malloc(out_sz, 0)
    ds_i = acl.mdl.create_dataset()
    b = acl.create_data_buffer(dev_in, in_sz); b = b[0] if isinstance(b, (tuple, list)) else b
    acl.mdl.add_dataset_buffer(ds_i, b)
    ds_o = acl.mdl.create_dataset()
    b = acl.create_data_buffer(dev_out, out_sz); b = b[0] if isinstance(b, (tuple, list)) else b
    acl.mdl.add_dataset_buffer(ds_o, b)
    stream, _ = acl.rt.create_stream()
    for _ in range(20): acl.mdl.execute_async(model_id, ds_i, ds_o, stream)
    acl.rt.synchronize_stream(stream)
    t0 = time.perf_counter()
    for _ in range(n): acl.mdl.execute_async(model_id, ds_i, ds_o, stream)
    t_submit = (time.perf_counter()-t0)/n*1e6
    acl.rt.synchronize_stream(stream)
    total = (time.perf_counter()-t0)/n*1000
    q.put({'chip': global_chip, 'submit_us': t_submit, 'total_ms': total})

if __name__ == '__main__':
    N = 2000
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(c, N, q)) for c in range(8)]
    [p.start() for p in ps]
    rs = [q.get() for _ in range(8)]
    [p.join() for p in ps]
    for r in sorted(rs, key=lambda x: x['chip']):
        print(f"  chip{r['chip']}: submit={r['submit_us']:.0f}us  {1000/r['total_ms']:.0f} fps")
    total = sum(1000/r['total_ms'] for r in rs)
    print(f"8 芯片合计: {total:.0f} fps  (单芯片 3263fps 基线 → 扩展比 {total/3263:.2f}x)")
