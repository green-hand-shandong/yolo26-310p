#!/usr/bin/env python3
"""310P 视频文件端到端链路：FFmpeg(CPU) 解码 -> DVPP VPC -> OM 推理（事件流水）

背景：CANN 8.2 的 pyACL vdec_send_frame 签名为 5 参数（"kkkkO"，反汇编确认），
但 C 层持续返回 100000（参数无效），判定为 8.2.RC1 在 310P3 上的 VDEC 运行时
组件不完整，归入 CANN 9.x 升级后待办。本脚本用 OpenCV/FFmpeg CPU 解码替代。

链路: h264 文件 --cv2 解码+I420→NV12--> H2D(dvpp_malloc) --VPC resize+CSC(fmt12)--> OM(AIPP)
      双缓冲 + event 流水，DVPP 与 AI Core 并行。

踩坑补充：
- VPC 输入 buffer 同样必须 dvpp_malloc（rt.malloc 输出/输入都会走慢速路径或全 0）
- cv2.COLOR_BGR2YUV_I420 是 YUV420P(3平面)，DVPP 要 NV12(2平面)，需手工交错 UV

运行: source set_env.sh 后 python3 video_e2e_test.py [h264路径] [帧数]
"""
import cv2, numpy as np, time
import acl, threading

acl.init(); acl.rt.set_device(0)
ctx, _ = acl.rt.create_context(0)
pp_stream, _ = acl.rt.create_stream()
infer_stream, _ = acl.rt.create_stream()
ev, _ = acl.rt.create_event()
chn = acl.media.dvpp_create_channel_desc()
assert acl.media.dvpp_create_channel(chn) == 0

model_id, _ = acl.mdl.load_from_file('yolo26n_e2e_aipp.om')
model_desc = acl.mdl.create_desc(); acl.mdl.get_desc(model_desc, model_id)
in_size = acl.mdl.get_input_size_by_index(model_desc, 0)
out_size = acl.mdl.get_output_size_by_index(model_desc, 0)

W = H = 640
STRIDE_W, STRIDE_H = (W+15)//16*16, (H+1)//2*2
RGB_STRIDE = (W*3+15)//16*16
rs_cfg = acl.media.dvpp_create_resize_config()
acl.media.dvpp_set_resize_config_interpolation(rs_cfg, 0)

NB = 2
buffers = []
for _ in range(NB):
    dev_yuv, _ = acl.media.dvpp_malloc(STRIDE_W*STRIDE_H*3//2)
    dev_rgb, _ = acl.media.dvpp_malloc(RGB_STRIDE*H)
    dev_out, _ = acl.rt.malloc(out_size, 0)
    pic_yuv = acl.media.dvpp_create_pic_desc()
    for s, v in [(acl.media.dvpp_set_pic_desc_data, dev_yuv), (acl.media.dvpp_set_pic_desc_size, STRIDE_W*STRIDE_H*3//2),
                 (acl.media.dvpp_set_pic_desc_width, W), (acl.media.dvpp_set_pic_desc_height, H),
                 (acl.media.dvpp_set_pic_desc_width_stride, STRIDE_W), (acl.media.dvpp_set_pic_desc_height_stride, STRIDE_H),
                 (acl.media.dvpp_set_pic_desc_format, 1)]: s(pic_yuv, v)
    pic_rgb = acl.media.dvpp_create_pic_desc()
    for s, v in [(acl.media.dvpp_set_pic_desc_data, dev_rgb), (acl.media.dvpp_set_pic_desc_size, RGB_STRIDE*H),
                 (acl.media.dvpp_set_pic_desc_width, W), (acl.media.dvpp_set_pic_desc_height, H),
                 (acl.media.dvpp_set_pic_desc_width_stride, RGB_STRIDE), (acl.media.dvpp_set_pic_desc_height_stride, H),
                 (acl.media.dvpp_set_pic_desc_format, 12)]: s(pic_rgb, v)
    ds_in = acl.mdl.create_dataset()
    bb = acl.create_data_buffer(dev_rgb, RGB_STRIDE*H); bb = bb[0] if isinstance(bb,(tuple,list)) else bb
    acl.mdl.add_dataset_buffer(ds_in, bb)
    ds_out = acl.mdl.create_dataset()
    bb = acl.create_data_buffer(dev_out, out_size); bb = bb[0] if isinstance(bb,(tuple,list)) else bb
    acl.mdl.add_dataset_buffer(ds_out, bb)
    buffers.append(dict(dev_yuv=dev_yuv, dev_rgb=dev_rgb, dev_out=dev_out, pic_yuv=pic_yuv, pic_rgb=pic_rgb, ds_in=ds_in, ds_out=ds_out))

cap = cv2.VideoCapture('/tmp/test_1080p.h264')
assert cap.isOpened(), "open h264 failed"

def frame_to_yuv(frame_bgr):
    # cv2 I420 是 YUV420P(3平面)；DVPP 要 NV12(Y+UV交错)
    W, H = 640, 640
    yuv = cv2.cvtColor(cv2.resize(frame_bgr, (W, H)), cv2.COLOR_BGR2YUV_I420).ravel()
    y = yuv[:W*H]; u = yuv[W*H:W*H+W*H//4]; v = yuv[W*H+W*H//4:]
    uv = np.empty(W*H//2, dtype=np.uint8)
    uv[0::2] = u; uv[1::2] = v
    return np.concatenate([y, uv]).tobytes()

n_frames = 0
t0 = time.perf_counter()
last_results = None
while n_frames < 60:
    ok, frame = cap.read()
    if not ok: break
    b = buffers[n_frames % NB]
    yuv = frame_to_yuv(frame)
    # H2D 帧 YUV（dvpp_malloc —— VPC 输入必须 DVPP 内存）
    dev_tmp, _ = acl.media.dvpp_malloc(len(yuv))
    acl.rt.memcpy(dev_tmp, len(yuv), acl.util.numpy_to_ptr(np.frombuffer(yuv, np.uint8)), len(yuv), 1)
    # VPC resize+CSC（异步）—— 输入 desc 指向本帧数据
    acl.media.dvpp_set_pic_desc_data(b['pic_yuv'], dev_tmp)
    acl.media.dvpp_set_pic_desc_size(b['pic_yuv'], len(yuv))
    acl.media.dvpp_vpc_resize_async(chn, b['pic_yuv'], b['pic_rgb'], rs_cfg, pp_stream)
    acl.rt.record_event(ev, pp_stream)
    acl.rt.stream_wait_event(infer_stream, ev)
    acl.mdl.execute_async(model_id, b['ds_in'], b['ds_out'], infer_stream)
    acl.rt.synchronize_stream(infer_stream)
    acl.rt.free(dev_tmp)
    n_frames += 1
acl.rt.synchronize_stream(pp_stream)
dt = (time.perf_counter()-t0)/max(n_frames,1)*1000
print(f"FFmpeg(CPU解码)+DVPP+OM 流水: {dt:.2f} ms/帧 = {1000/dt:.0f} fps  ({n_frames} 帧)")

# 数值验证：最后一帧
b = buffers[(n_frames-1) % NB]
host_out = np.empty(out_size//4, dtype=np.float32)
acl.rt.memcpy(host_out.ctypes.data, out_size, b['dev_out'], out_size, 2)
arr = host_out.reshape(1, 300, 6)[0]
COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]
hi = arr[arr[:,4] > 0.5]
print(f"末帧检出: {len(hi)} 个")
for row in hi[:5]:
    cls = COCO[int(row[5])] if int(row[5]) < len(COCO) else int(row[5])
    print(f"  {cls:<10} {row[4]:.3f}")
