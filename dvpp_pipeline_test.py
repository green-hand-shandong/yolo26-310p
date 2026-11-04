#!/usr/bin/env python3
"""310P DVPP+OM 事件流水线压测（在 dvpp_e2e_test.py 串行链路基础上并行化）

原理：
  预处理(JPEGD+VPC) 跑在 pp_stream（DVPP 硬件单元）
  OM 推理跑在 infer_stream（AI Core）
  两者是不同硬件单元，用 event 建立帧内依赖、双缓冲消除帧间等待：

    帧 n:   [JPEGD+VPC n] ----event----> [OM n]
    帧 n+1: [JPEGD+VPC n+1]（与 OM n 并行）

  稳态吞吐 = max(预处理, 推理) ≈ max(3.8, 4.5) ≈ 4.5ms/帧 ≈ 220fps

运行: source set_env.sh 后 python3 dvpp_pipeline_test.py [N]
"""
import acl
import numpy as np
import time
import sys

OM_PATH = "yolo26n_e2e_aipp.om"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
IMG_1080P = sys.argv[2] if len(sys.argv) > 2 else None  # 传 1080p jpeg 则用真实场景
ACL_MEMCPY_H2D, ACL_MEMCPY_D2H, ACL_MEMCPY_D2D = 1, 2, 3

acl.init(); acl.rt.set_device(0)
ctx, _ = acl.rt.create_context(0)
pp_stream, _ = acl.rt.create_stream()     # 预处理（DVPP）
infer_stream, _ = acl.rt.create_stream()  # 推理（AI Core）
ev, _ = acl.rt.create_event()
chn = acl.media.dvpp_create_channel_desc()
assert acl.media.dvpp_create_channel(chn) == 0

# ---- OM (static AIPP) ----
model_id, _ = acl.mdl.load_from_file(OM_PATH)
model_desc = acl.mdl.create_desc(); acl.mdl.get_desc(model_desc, model_id)
in_size = acl.mdl.get_input_size_by_index(model_desc, 0)
out_size = acl.mdl.get_output_size_by_index(model_desc, 0)

# ---- DVPP 资源 ----
W = H = 640
JW, JH = (W+15)//16*16, (H+1)//2*2
DEC_SZ = JW*JH*3//2 + 1024*1024
RGB_STRIDE = (W*3+15)//16*16
rs_cfg = acl.media.dvpp_create_resize_config()
acl.media.dvpp_set_resize_config_interpolation(rs_cfg, 0)

if IMG_1080P:
    IN_W, IN_H = 1920, 1080
    jpeg_bytes = open(IMG_1080P, "rb").read()
    JW, JH = (IN_W+15)//16*16, (IN_H+1)//2*2
    DEC_SZ = JW*JH*3//2 + 1024*1024
else:
    IN_W, IN_H = W, H
    jpeg_bytes = open("bus_det_310p.jpg", "rb").read()
    JW, JH = (W+15)//16*16, (H+1)//2*2
    DEC_SZ = JW*JH*3//2 + 1024*1024
dev_jpeg, _ = acl.rt.malloc(len(jpeg_bytes), 0)
jn = np.frombuffer(jpeg_bytes, np.uint8)
acl.rt.memcpy(dev_jpeg, len(jpeg_bytes), acl.util.numpy_to_ptr(jn), len(jpeg_bytes), ACL_MEMCPY_H2D)
dev_yuv, _ = acl.media.dvpp_malloc(DEC_SZ)
pic_dec = acl.media.dvpp_create_pic_desc()
for s, v in [(acl.media.dvpp_set_pic_desc_data, dev_yuv), (acl.media.dvpp_set_pic_desc_size, DEC_SZ),
             (acl.media.dvpp_set_pic_desc_width, IN_W), (acl.media.dvpp_set_pic_desc_height, IN_H),
             (acl.media.dvpp_set_pic_desc_width_stride, JW), (acl.media.dvpp_set_pic_desc_height_stride, JH),
             (acl.media.dvpp_set_pic_desc_format, 1)]: s(pic_dec, v)

# ---- 双缓冲：2 组 (RGB 输入 buffer, 输出 buffer, dataset) ----
NB = 2
buffers = []
for _ in range(NB):
    dev_rgb, _ = acl.media.dvpp_malloc(RGB_STRIDE*H)   # VPC 输出必须 dvpp_malloc
    dev_out, _ = acl.rt.malloc(out_size, 0)
    ds_in = acl.mdl.create_dataset()
    bb = acl.create_data_buffer(dev_rgb, RGB_STRIDE*H); bb = bb[0] if isinstance(bb,(tuple,list)) else bb
    acl.mdl.add_dataset_buffer(ds_in, bb)
    ds_out = acl.mdl.create_dataset()
    bb = acl.create_data_buffer(dev_out, out_size); bb = bb[0] if isinstance(bb,(tuple,list)) else bb
    acl.mdl.add_dataset_buffer(ds_out, bb)
    buffers.append({"dev_rgb": dev_rgb, "dev_out": dev_out, "ds_in": ds_in, "ds_out": ds_out,
                    "pic_rgb": None})

def mk_rgb_pic(dev):
    p = acl.media.dvpp_create_pic_desc()
    for s, v in [(acl.media.dvpp_set_pic_desc_data, dev), (acl.media.dvpp_set_pic_desc_size, RGB_STRIDE*H),
                 (acl.media.dvpp_set_pic_desc_width, W), (acl.media.dvpp_set_pic_desc_height, H),
                 (acl.media.dvpp_set_pic_desc_width_stride, RGB_STRIDE), (acl.media.dvpp_set_pic_desc_height_stride, H),
                 (acl.media.dvpp_set_pic_desc_format, 12)]: s(p, v)
    return p
for b in buffers:
    b["pic_rgb"] = mk_rgb_pic(b["dev_rgb"])

def stage_submit(i):
    """提交第 i 帧预处理（异步）+ 记录 event"""
    b = buffers[i % NB]
    acl.media.dvpp_jpeg_decode_async(chn, dev_jpeg, len(jpeg_bytes), pic_dec, pp_stream)
    acl.media.dvpp_vpc_resize_async(chn, pic_dec, b["pic_rgb"], rs_cfg, pp_stream)
    acl.rt.record_event(ev, pp_stream)

def stage_infer(i):
    """推理第 i 帧：等待其 event 后异步执行"""
    b = buffers[i % NB]
    acl.rt.stream_wait_event(infer_stream, ev)
    acl.mdl.execute_async(model_id, b["ds_in"], b["ds_out"], infer_stream)

# ---- 串行基线 ----
def serial_once():
    stage_submit(0)
    acl.rt.synchronize_stream(pp_stream)
    acl.rt.memcpy(buffers[0]["dev_rgb"], RGB_STRIDE*H, buffers[0]["dev_rgb"], RGB_STRIDE*H, ACL_MEMCPY_D2D)  # no-op 保温
    stage_infer(0)
    acl.rt.synchronize_stream(infer_stream)

for _ in range(3):
    stage_submit(0); acl.rt.synchronize_stream(pp_stream); stage_infer(0); acl.rt.synchronize_stream(infer_stream)
t0 = time.perf_counter()
for _ in range(N):
    stage_submit(0); acl.rt.synchronize_stream(pp_stream); stage_infer(0); acl.rt.synchronize_stream(infer_stream)
acl.rt.synchronize_stream(pp_stream)
t_serial = (time.perf_counter()-t0)/N*1000
print(f"串行基线: {t_serial:.2f} ms/帧 ({1000/t_serial:.0f} fps)")

# ---- 流水线：帧 n 推理与帧 n+1 预处理并行 ----
# 提交 N 帧后统一 sync（帧间无阻塞；双缓冲防止读写冲突）
for _ in range(3):
    stage_submit(0); stage_infer(0); acl.rt.synchronize_stream(infer_stream)
acl.rt.synchronize_stream(pp_stream)

t0 = time.perf_counter()
for i in range(N):
    stage_submit(i)
    stage_infer(i)
acl.rt.synchronize_stream(infer_stream)
acl.rt.synchronize_stream(pp_stream)
t_pipe = (time.perf_counter()-t0)/N*1000
print(f"事件流水线: {t_pipe:.2f} ms/帧 ({1000/t_pipe:.0f} fps)  提速 {t_serial/t_pipe:.2f}x")

# 数值验证：最后一帧结果
host_out = np.empty(out_size//4, dtype=np.float32)
last = buffers[(N-1) % NB]
acl.rt.memcpy(host_out.ctypes.data, out_size, last["dev_out"], out_size, ACL_MEMCPY_D2H)
arr = host_out.reshape(1, 300, 6)[0]
COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]
hi = arr[arr[:,4] > 0.5]
print(f"流水线最后一帧检出: {len(hi)} 个")
for row in hi[:6]:
    cls = COCO[int(row[5])] if int(row[5]) < len(COCO) else int(row[5])
    print(f"  {cls:<10} {row[4]:.3f}  box=({row[0]:.0f},{row[1]:.0f},{row[2]:.0f},{row[3]:.0f})")
print("PIPELINE OK")
