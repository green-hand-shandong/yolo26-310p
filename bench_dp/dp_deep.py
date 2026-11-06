"""深挖 submit 295us 的构成，并验证三种代码级优化路径"""
import sys, os, time
import multiprocessing as mp

def worker_py(args):
    """A: pyACL 基线（已有数据 295us）"""
    chip, n, q = args
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(chip)
    import acl, numpy as np
    acl.init(); acl.rt.set_device(0)
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
    # batch 提交：一次 execute 换成 bs8 看单位图片提交开销
    t0 = time.perf_counter()
    for _ in range(n): acl.mdl.execute_async(model_id, ds_i, ds_o, stream)
    t_submit = (time.perf_counter()-t0)/n
    acl.rt.synchronize_stream(stream)
    q.put(('py-bs1', t_submit*1e6))

def worker_batch(args):
    """B: batch 摊薄——bs8 OM 一次提交 8 张"""
    chip, n, q = args
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(chip)
    import acl, numpy as np
    acl.init(); acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)
    model_id, _ = acl.mdl.load_from_file('/work/yolo26n_cann901_bs8.om')
    md = acl.mdl.create_desc(); acl.mdl.get_desc(md, model_id)
    in_sz = acl.mdl.get_input_size_by_index(md, 0)
    out_sz = acl.mdl.get_output_size_by_index(md, 0)
    one = np.fromfile('/work/input_1x3x640x640.bin', dtype=np.float32)
    np_in = np.tile(one, 8)
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
    for _ in range(10): acl.mdl.execute_async(model_id, ds_i, ds_o, stream)
    acl.rt.synchronize_stream(stream)
    t0 = time.perf_counter()
    for _ in range(n): acl.mdl.execute_async(model_id, ds_i, ds_o, stream)
    t_submit = (time.perf_counter()-t0)/n
    acl.rt.synchronize_stream(stream)
    # 每"图片"提交开销 = submit/8
    q.put(('py-bs8', t_submit*1e6/8))

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    N = 3000
    jobs = [(0, N, q), (0, N, q)]
    ps = [ctx.Process(target=worker_py, args=(j,)) for j in jobs]
    [p.start() for p in ps]; rs = [q.get() for _ in ps]; [p.join() for p in ps]
    print(f"A py-bs1 双芯片并发提交: {rs[0][1]:.0f} + {rs[1][1]:.0f} us/img")
    ps = [ctx.Process(target=worker_batch, args=(j,)) for j in jobs]
    [p.start() for p in ps]; rs = [q.get() for _ in ps]; [p.join() for p in ps]
    print(f"B py-bs8 双芯片并发提交(摊到每图): {rs[0][1]:.0f} + {rs[1][1]:.0f} us/img")
