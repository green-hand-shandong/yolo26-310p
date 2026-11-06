"""分相测量：execute 提交耗时 vs 同步等待耗时（区分 host 驱动开销 vs device 执行）"""
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
    stream, _ = acl.rt.create_stream()
    for _ in range(20):
        acl.mdl.execute(model_id, ds_i, ds_o)
    acl.rt.synchronize_stream(stream)
    # 分相：异步提交总耗时（host 侧）+ 末尾同步（device 排空）
    t_submit = 0.0
    t0 = time.perf_counter()
    for _ in range(n):
        ts = time.perf_counter()
        acl.mdl.execute_async(model_id, ds_i, ds_o, stream)
        t_submit += time.perf_counter() - ts
    t_sync0 = time.perf_counter()
    acl.rt.synchronize_stream(stream)
    t_sync = time.perf_counter() - t_sync0
    total = time.perf_counter() - t0
    q.put({'tag': tag, 'submit_us': t_submit/n*1e6, 'sync_ms': t_sync*1000,
           'total_ms': total/n*1000})

if __name__ == '__main__':
    N = 3000
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(0, N, q, 'single')); p.start()
    r0 = q.get(); p.join()
    p1 = ctx.Process(target=worker, args=(0, N, q, 'dualA')); p1.start()
    p2 = ctx.Process(target=worker, args=(1, N, q, 'dualB')); p2.start()
    r1 = q.get(); r2 = q.get(); p1.join(); p2.join()
    print(f"单芯片: submit={r0['submit_us']:.1f}us/it  sync_total={r0['total_ms']:.3f} ms/img")
    print(f"双芯片: A submit={r1['submit_us']:.1f}us total={r1['total_ms']:.3f} ms | B submit={r2['submit_us']:.1f}us total={r2['total_ms']:.3f} ms")
    print(f"提交开销占比（单）: {r0['submit_us']/(r0['total_ms']*1000)*100:.0f}%")
