"""拓扑对照：同卡双芯 vs 跨卡双芯 vs 同芯片双进程"""
import sys, os, time
import multiprocessing as mp

# davinci 全局编号（npu-smi 的 Device 列）
# 卡7(03:00.0): davinci0,1 | 卡5(04:00.0): davinci2,3
# 卡1(06:00.0): davinci4,5 | 卡3(07:00.0): davinci6,7

def worker(global_chip, n, q, tag):
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
    q.put({'tag': tag, 'submit_us': t_submit, 'total_ms': total})

if __name__ == '__main__':
    N = 3000
    ctx = mp.get_context('spawn')
    q = ctx.Queue()

    def run2(g1, g2, tag):
        p1 = ctx.Process(target=worker, args=(g1, N, q, tag+'-A')); p1.start()
        p2 = ctx.Process(target=worker, args=(g2, N, q, tag+'-B')); p2.start()
        r1 = q.get(); r2 = q.get(); p1.join(); p2.join()
        return r1, r2

    # 基线：单芯片
    p = ctx.Process(target=worker, args=(4, N, q, 'single')); p.start()
    r0 = q.get(); p.join()
    print(f"单芯片: submit={r0['submit_us']:.0f}us total={r0['total_ms']:.3f} ms ({1000/r0['total_ms']:.0f} fps)")

    # 场景1：同卡双芯（卡1: davinci4+5，PCIe 06:00.0）
    r1, r2 = run2(4, 5, 'same-card')
    s1 = 1000/r1['total_ms'] + 1000/r2['total_ms']
    print(f"同卡双芯(4+5):   submit={r1['submit_us']:.0f}/{r2['submit_us']:.0f}us  合计 {s1:.0f} fps  扩展比 {s1/(1000/r0['total_ms']):.2f}x")

    # 场景2：跨卡双芯（卡1 chip4 + 卡5 chip2，PCIe 06:00.0 + 04:00.0）
    r1, r2 = run2(4, 2, 'cross-card')
    s2 = 1000/r1['total_ms'] + 1000/r2['total_ms']
    print(f"跨卡双芯(4+2):   submit={r1['submit_us']:.0f}/{r2['submit_us']:.0f}us  合计 {s2:.0f} fps  扩展比 {s2/(1000/r0['total_ms']):.2f}x")

    # 场景3：最远跨卡（卡7 chip0 + 卡3 chip6）
    r1, r2 = run2(0, 6, 'far-cross')
    s3 = 1000/r1['total_ms'] + 1000/r2['total_ms']
    print(f"远跨卡(0+6):     submit={r1['submit_us']:.0f}/{r2['submit_us']:.0f}us  合计 {s3:.0f} fps  扩展比 {s3/(1000/r0['total_ms']):.2f}x")
