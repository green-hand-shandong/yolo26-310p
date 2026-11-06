"""分层定位：同芯片双进程 vs 双芯片双进程"""
import sys, os, time
import multiprocessing as mp

def worker(chip, n, q, tag):
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(chip)
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
    for _ in range(20):
        acl.mdl.execute(model_id, ds_i, ds_o)
    acl.rt.synchronize_stream(0)
    t0 = time.perf_counter()
    for _ in range(n):
        acl.mdl.execute(model_id, ds_i, ds_o)
    acl.rt.synchronize_stream(0)
    q.put({'tag': tag, 'chip': chip, 'ms': (time.perf_counter()-t0)/n*1000})

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    # A: 双进程同芯片0
    pa = ctx.Process(target=worker, args=(0, N, q, 'same-chip-A')); pa.start()
    pb = ctx.Process(target=worker, args=(0, N, q, 'same-chip-B')); pb.start()
    ra = q.get(); rb = q.get(); pa.join(); pb.join()
    # B: 双进程双芯片
    pc = ctx.Process(target=worker, args=(0, N, q, 'dual-chip-A')); pc.start()
    pd = ctx.Process(target=worker, args=(1, N, q, 'dual-chip-B')); pd.start()
    rc = q.get(); rd = q.get(); pc.join(); pd.join()
    # C: 单进程单芯片基线
    pe = ctx.Process(target=worker, args=(0, N, q, 'single')); pe.start()
    re = q.get(); pe.join()
    print(f"单进程单芯片:  {re['ms']:.3f} ms/img = {1000/re['ms']:.0f} fps")
    print(f"双进程同芯片:  A={ra['ms']:.3f} B={rb['ms']:.3f}  合计 {1000/ra['ms']+1000/rb['ms']:.0f} fps")
    print(f"双进程双芯片:  A={rc['ms']:.3f} B={rd['ms']:.3f}  合计 {1000/rc['ms']+1000/rd['ms']:.0f} fps")
