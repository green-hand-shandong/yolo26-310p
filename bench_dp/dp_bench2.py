"""DP 基准 v2：真实输入 + 交错验证 + 更长迭代"""
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
    # 分桶测量：每 100 it 记录一次，看稳态
    buckets = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n // 5):
            acl.mdl.execute(model_id, ds_i, ds_o)
        acl.rt.synchronize_stream(0)
        buckets.append((time.perf_counter() - t0) / (n / 5) * 1000)
    q.put({'chip': chip, 'tag': tag, 'buckets_ms': buckets})

if __name__ == '__main__':
    chip0, chip1 = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(chip0, N, q, 'single')); p.start()
    r0 = q.get(); p.join()
    p1 = ctx.Process(target=worker, args=(chip0, N, q, 'dual')); p1.start()
    p2 = ctx.Process(target=worker, args=(chip1, N, q, 'dual')); p2.start()
    r1 = q.get(); r2 = q.get(); p1.join(); p2.join()
    s_ms = min(r0['buckets_ms'])
    print(f"单芯片({r0['chip']}) 稳态: {s_ms:.3f} ms/img = {1000/s_ms:.0f} fps")
    print(f"  单芯片分桶: {[f'{b:.2f}' for b in r0['buckets_ms']]}")
    print(f"双芯片 chip{r1['chip']}: {[f'{b:.2f}' for b in r1['buckets_ms']]}")
    print(f"双芯片 chip{r2['chip']}: {[f'{b:.2f}' for b in r2['buckets_ms']]}")
    d1, d2 = min(r1['buckets_ms']), min(r2['buckets_ms'])
    total_fps = 1000/d1 + 1000/d2
    print(f"双芯片合计: {total_fps:.0f} fps  扩展比: {(1000/d1 + 1000/d2)/(1000/s_ms):.3f}x")
