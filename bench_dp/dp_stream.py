"""流水化提交：多 stream 异步提交 + 末尾统一同步，掩盖 host 提交开销"""
import sys, os, time
import multiprocessing as mp

def worker(chip, n_streams, n, q, tag):
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
    streams, ds_is, ds_os = [], [], []
    for _ in range(n_streams):
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
    for s in streams:
        for _ in range(10):
            acl.mdl.execute_async(model_id, ds_is[0], ds_os[0], s)
    for s in streams:
        acl.rt.synchronize_stream(s)
    # 流水测量：n_streams 个 stream 轮转异步提交
    iters = n * n_streams
    t0 = time.perf_counter()
    for i in range(iters):
        acl.mdl.execute_async(model_id, ds_is[i % n_streams], ds_os[i % n_streams], streams[i % n_streams])
    for s in streams:
        acl.rt.synchronize_stream(s)
    total = time.perf_counter() - t0
    q.put({'tag': tag, 'streams': n_streams, 'total_ms': total/iters*1000,
           'fps': iters/total})

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    N = 800
    results = {}
    for tag, chip, ns in [('single-1s', 0, 1), ('single-4s', 0, 4), ('dual-2s', None, 2), ('dual-4s', None, 4)]:
        if chip is None:
            ps = [ctx.Process(target=worker, args=(c, ns, N, q, tag)) for c in (0, 1)]
        else:
            ps = [ctx.Process(target=worker, args=(chip, ns, N, q, tag))]
        [p.start() for p in ps]
        rs = [q.get() for _ in ps]
        [p.join() for p in ps]
        total_fps = sum(r['fps'] for r in rs)
        results[tag] = total_fps
        print(f"{tag}: 合计 {total_fps:.0f} fps  (每流 {rs[0]['total_ms']:.3f} ms/it)")
    print(f"\n扩展比 vs single-1s: dual-2s={results['dual-2s']/results['single-1s']:.2f}x  dual-4s={results['dual-4s']/results['single-1s']:.2f}x")
