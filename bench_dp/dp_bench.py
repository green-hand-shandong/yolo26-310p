"""DP 双芯片吞吐实测：单芯片 vs 双芯片（每芯片独立进程跑同一推理负载）
负载：yolo26n_cann901_bs1.om fp16 bs1 同步推理（纯 AI Core，无 DVPP）
用法：python3 dp_bench.py <chip0> <chip1> <N>
"""
import sys, os, time, json
import multiprocessing as mp

def worker(chip, n, q):
    """单芯片 worker：加载模型跑 n 次推理，回报耗时"""
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(chip)
    import acl, numpy as np
    ret = acl.init()
    acl.rt.set_device(0)  # 每进程只见自己芯片，逻辑设备 0
    ctx, _ = acl.rt.create_context(0)
    model_id, _ = acl.mdl.load_from_file('/work/yolo26n_cann901_bs1.om')
    md = acl.mdl.create_desc(); acl.mdl.get_desc(md, model_id)
    in_sz = acl.mdl.get_input_size_by_index(md, 0)
    out_sz = acl.mdl.get_output_size_by_index(md, 0)
    np_in = np.random.rand(in_sz // 4).astype(np.float32)
    dev_in, _ = acl.rt.malloc(in_sz, 0)
    acl.rt.memcpy(dev_in, in_sz, acl.util.numpy_to_ptr(np_in), in_sz, 1)
    dev_out, _ = acl.rt.malloc(out_sz, 0)
    ds_i = acl.mdl.create_dataset()
    b = acl.create_data_buffer(dev_in, in_sz); b = b[0] if isinstance(b, (tuple, list)) else b
    acl.mdl.add_dataset_buffer(ds_i, b)
    ds_o = acl.mdl.create_dataset()
    b = acl.create_data_buffer(dev_out, out_sz); b = b[0] if isinstance(b, (tuple, list)) else b
    acl.mdl.add_dataset_buffer(ds_o, b)
    # warmup
    for _ in range(10):
        acl.mdl.execute(model_id, ds_i, ds_o)
    acl.rt.synchronize_stream(0)
    t0 = time.perf_counter()
    for _ in range(n):
        acl.mdl.execute(model_id, ds_i, ds_o)
    acl.rt.synchronize_stream(0)
    dt = time.perf_counter() - t0
    q.put({'chip': chip, 'n': n, 'sec': dt, 'fps': n / dt})

if __name__ == '__main__':
    chip0, chip1 = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    # 1) 单芯片基线
    p = ctx.Process(target=worker, args=(chip0, N, q)); p.start()
    r0 = q.get(); p.join()
    # 2) 双芯片并发
    p1 = ctx.Process(target=worker, args=(chip0, N, q)); p1.start()
    p2 = ctx.Process(target=worker, args=(chip1, N, q)); p2.start()
    r1 = q.get(); r2 = q.get(); p1.join(); p2.join()
    single = r0['fps']
    dual_each = min(r1['fps'], r2['fps'])
    dual_total = r1['fps'] + r2['fps']
    print(f"单芯片({r0['chip']}): {r0['fps']:.1f} fps  ({r0['sec']/r0['n']*1000:.2f} ms/img)")
    print(f"双芯片并发: chip{r1['chip']}={r1['fps']:.1f} fps, chip{r2['chip']}={r2['fps']:.1f} fps")
    print(f"双芯片合计: {dual_total:.1f} fps  扩展比: {dual_total/single:.3f}x (理想 2.000)")
    print(f"较慢芯片效率: {dual_each/single*100:.1f}%")
