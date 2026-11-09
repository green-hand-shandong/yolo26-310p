"""ATC op_select_implmode 对比：默认 vs high_performance（NPU 实测延迟）"""
import acl, numpy as np, time, sys

def bench(om_path, n=2000):
    acl.init()
    acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)
    model_id, ret = acl.mdl.load_from_file(om_path)
    md = acl.mdl.create_desc(); acl.mdl.get_desc(md, model_id)
    in_sz = acl.mdl.get_input_size_by_index(md, 0)
    out_sz = acl.mdl.get_output_size_by_index(md, 0)
    np_in = np.fromfile('/data2/gxw/yolo26-310p-eval/input_1x3x640x640.bin', dtype=np.float32)
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
    for _ in range(50): acl.mdl.execute_async(model_id, ds_i, ds_o, stream)
    acl.rt.synchronize_stream(stream)
    t0 = time.perf_counter()
    for _ in range(n): acl.mdl.execute_async(model_id, ds_i, ds_o, stream)
    acl.rt.synchronize_stream(stream)
    dt = (time.perf_counter()-t0)/n*1000
    acl.mdl.unload(model_id); acl.rt.destroy_context(ctx); acl.finalize()
    return dt

d1 = bench('/data2/gxw/yolo26-310p-eval/yolo26n_det_default.om')
d2 = bench('/data2/gxw/yolo26-310p-eval/yolo26n_det_hp.om')
print(f"默认模式:  {d1:.3f} ms/img ({1000/d1:.0f} fps)")
print(f"high_perf: {d2:.3f} ms/img ({1000/d2:.0f} fps)  提速 {d1/d2:.2f}x")
