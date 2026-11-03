"""端到端: DVPP 硬件 resize+CSC(YUV→BGR) → NPU 推理, 全硬件预处理"""
import acl
import numpy as np
import time
import threading
import queue
import sys

acl.init(); acl.rt.set_device(0)
ctx_main, _ = acl.rt.create_context(0)

OM = sys.argv[1] if len(sys.argv)>1 else "/work/yolo26n_e2e_310p.om"
N = int(sys.argv[2]) if len(sys.argv)>2 else 100
W,H = 640,640
IN_W,IN_H = 1280,720
OUT_SZ = (W*3+15)//16*16 * H  # stride 对齐16
STRIDE_W = (W*3+15)//16*16

# 加载模型
mid,_ = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc(); acl.mdl.get_desc(d, mid)
in_sz = acl.mdl.get_input_size_by_index(d, 0)
out_sz = acl.mdl.get_output_size_by_index(d, 0)

# DVPP (主线程创建, 主线程使用)
chn = acl.media.dvpp_create_channel_desc()
acl.media.dvpp_create_channel(chn)
in_y_sz = IN_W*IN_H; in_uv = IN_W*IN_H//2
dev_frame, r = acl.media.dvpp_malloc(in_y_sz+in_uv)
frame = np.random.randint(0,255,in_y_sz+in_uv,dtype=np.uint8)
acl.rt.memcpy(dev_frame, in_y_sz+in_uv, frame.ctypes.data, in_y_sz+in_uv, 1)
pic_in = acl.media.dvpp_create_pic_desc()
for setter, val in [(acl.media.dvpp_set_pic_desc_data, dev_frame),
    (acl.media.dvpp_set_pic_desc_size, in_y_sz+in_uv),
    (acl.media.dvpp_set_pic_desc_width, IN_W), (acl.media.dvpp_set_pic_desc_height, IN_H),
    (acl.media.dvpp_set_pic_desc_width_stride, IN_W),
    (acl.media.dvpp_set_pic_desc_height_stride, IN_H),
    (acl.media.dvpp_set_pic_desc_format, 1)]:
    setter(pic_in, val)
dev_rst, r = acl.media.dvpp_malloc(OUT_SZ)
pic_out = acl.media.dvpp_create_pic_desc()
for setter, val in [(acl.media.dvpp_set_pic_desc_data, dev_rst),
    (acl.media.dvpp_set_pic_desc_size, OUT_SZ),
    (acl.media.dvpp_set_pic_desc_width, W), (acl.media.dvpp_set_pic_desc_height, H),
    (acl.media.dvpp_set_pic_desc_width_stride, STRIDE_W),
    (acl.media.dvpp_set_pic_desc_height_stride, H),
    (acl.media.dvpp_set_pic_desc_format, 12)]:  # BGR888
    setter(pic_out, val)
rs_cfg = acl.media.dvpp_create_resize_config()

# NPU
dev_npu_i, r = acl.rt.malloc(in_sz, 0); dev_npu_o, r = acl.rt.malloc(out_sz, 0)
ds_i = acl.mdl.create_dataset(); r1=acl.create_data_buffer(dev_npu_i,in_sz); acl.mdl.add_dataset_buffer(ds_i, r1[0] if isinstance(r1,(tuple,list)) else r1)
ds_o = acl.mdl.create_dataset(); r2=acl.create_data_buffer(dev_npu_o,out_sz); acl.mdl.add_dataset_buffer(ds_o, r2[0] if isinstance(r2,(tuple,list)) else r2)
h_f32 = np.empty(in_sz//4, dtype=np.float32)
h_out = np.empty(out_sz//4, dtype=np.float32)

def dvpp_resize_csc():
    """DVPP 硬件: resize + YUV→BGR 一步完成"""
    ret = acl.media.dvpp_vpc_resize_async(chn, pic_in, pic_out, rs_cfg, 0)
    if ret != 0: return None
    bgr = np.empty(OUT_SZ, dtype=np.uint8)
    acl.rt.memcpy(bgr.ctypes.data, OUT_SZ, dev_rst, OUT_SZ, 2)
    # 去 stride: 取每行前 W*3 字节
    return bgr[:H*STRIDE_W].reshape(H, STRIDE_W)[:, :W*3].reshape(H, W, 3)

def bgr_to_npu(bgr):
    """BGR uint8 → RGB float32/255 → NPU 输入"""
    h_f32[:] = bgr[::-1].transpose(2,0,1)[None].astype(np.float32).ravel() / 255.0
    acl.rt.memcpy(dev_npu_i, in_sz, acl.util.numpy_to_ptr(h_f32), in_sz, 1)
    acl.mdl.execute(mid, ds_i, ds_o)
    acl.rt.memcpy(acl.util.numpy_to_ptr(h_out), out_sz, dev_npu_o, out_sz, 2)
    return h_out.reshape(1,300,6)

# ===== 性能测量: 串联 DVPP CSC + NPU =====
for _ in range(5): dvpp_resize_csc()
t0 = time.perf_counter()
for _ in range(N):
    bgr = dvpp_resize_csc()
    det = bgr_to_npu(bgr)
dt = time.perf_counter()-t0
e2e_fps = N/dt
print(f"e2e (DVPP CSC + NPU 推理): {dt*1000/N:.2f} ms/帧 = {e2e_fps:.1f} fps")

# 拆分测量
t0 = time.perf_counter()
for _ in range(N): dvpp_resize_csc()
dt_dvpp = time.perf_counter()-t0
print(f"  DVPP resize+CSC only: {dt_dvpp*1000/N:.2f} ms/帧 ({N/dt_dvpp:.0f} fps)")
t0 = time.perf_counter()
for _ in range(N): bgr_to_npu(dvpp_resize_csc())
# 已含上面; 单独 NPU:
h_f32[:] = np.random.rand(*h_f32.shape)
for _ in range(5): acl.mdl.execute(mid, ds_i, ds_o)
t0 = time.perf_counter()
for _ in range(N):
    acl.rt.memcpy(dev_npu_i, in_sz, acl.util.numpy_to_ptr(h_f32), in_sz, 1)
    acl.mdl.execute(mid, ds_i, ds_o)
    acl.rt.memcpy(acl.util.numpy_to_ptr(h_out), out_sz, dev_npu_o, out_sz, 2)
dt_npu = time.perf_counter()-t0
print(f"  NPU 推理 only: {dt_npu*1000/N:.2f} ms/帧 ({N/dt_npu:.0f} fps)")
print(f"\n对比: 纯CPU YUV→RGB (np.repeat) = 14.01ms/帧 → DVPP CSC = {dt_dvpp*1000/N:.2f}ms ({14.01/(dt_dvpp*1000/N):.0f}x 加速)")
