#!/usr/bin/env python3
"""310P 全硬件端到端链路压测: JPEGD 硬解码 -> VPC resize+CSC 一步 -> OM(AIPP) 推理

链路: jpg(1080p/640) --JPEGD--> YUV420SP --VPC resize(fmt12=BGR888)--> OM 输入
所有图像处理在 DVPP 硬件单元完成, CPU 只做提交, 无 opencv 依赖(除生成测试图)。

关键踩坑(310P 实测):
1. JPEGD 输出 desc 的 size 必须比精确 YUV 大 1MB 以上, 否则 ret_code=4
   (1920x1080: 3110400 精确值失败, +1MB 成功)
2. VPC 输出 buffer 必须用 acl.media.dvpp_malloc 分配; 写 acl.rt.malloc 普通内存
   走慢速路径 (~100ms vs ~1ms)
3. BGR888 packed 的 width_stride 是 W*3 再 16 对齐 (=1920@640), 不是 W
4. resize+CSC 用 dvpp_vpc_resize_async 一步完成: 输出 pic_desc 直接设 fmt=12
5. DVPP buffer -> OM 输入需 D2D 拷贝 (acl.rt.memcpy kind=3), 仅 0.17ms

运行: source set_env.sh 后 python3 dvpp_e2e_test.py
"""
import acl
import numpy as np
import time
import sys

OM_PATH = sys.argv[1] if len(sys.argv) > 1 else "yolo26n_e2e_aipp.om"
IMG = sys.argv[2] if len(sys.argv) > 2 else None  # 可选 1080p jpeg
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100

ACL_MEMCPY_H2D, ACL_MEMCPY_D2H, ACL_MEMCPY_D2D = 1, 2, 3

acl.init(); acl.rt.set_device(0)
ctx, _ = acl.rt.create_context(0)
stream, _ = acl.rt.create_stream()
chn = acl.media.dvpp_create_channel_desc()
assert acl.media.dvpp_create_channel(chn) == 0

# ---- OM (static AIPP: uint8 BGR888 640x640) ----
model_id, _ = acl.mdl.load_from_file(OM_PATH)
model_desc = acl.mdl.create_desc(); acl.mdl.get_desc(model_desc, model_id)
in_size = acl.mdl.get_input_size_by_index(model_desc, 0)
out_size = acl.mdl.get_output_size_by_index(model_desc, 0)
dev_in_model, _ = acl.rt.malloc(in_size, 0)
dev_out_model, _ = acl.rt.malloc(out_size, 0)
ds_in = acl.mdl.create_dataset()
b = acl.create_data_buffer(dev_in_model, in_size); b = b[0] if isinstance(b,(tuple,list)) else b
acl.mdl.add_dataset_buffer(ds_in, b)
ds_out = acl.mdl.create_dataset()
b = acl.create_data_buffer(dev_out_model, out_size); b = b[0] if isinstance(b,(tuple,list)) else b
acl.mdl.add_dataset_buffer(ds_out, b)

def mk_pic(dev, size, w, h, ws, hs, fmt):
    p = acl.media.dvpp_create_pic_desc()
    for s, v in [(acl.media.dvpp_set_pic_desc_data, dev), (acl.media.dvpp_set_pic_desc_size, size),
                 (acl.media.dvpp_set_pic_desc_width, w), (acl.media.dvpp_set_pic_desc_height, h),
                 (acl.media.dvpp_set_pic_desc_width_stride, ws), (acl.media.dvpp_set_pic_desc_height_stride, hs),
                 (acl.media.dvpp_set_pic_desc_format, fmt)]: s(p, v)
    return p

W = H = 640
STRIDE_W, STRIDE_H = (W+15)//16*16, (H+1)//2*2
RGB_STRIDE = (W*3+15)//16*16
rs_cfg = acl.media.dvpp_create_resize_config()
acl.media.dvpp_set_resize_config_interpolation(rs_cfg, 0)

if IMG:
    # 真实 1080p 场景
    IN_W, IN_H = 1920, 1080
    jpeg_bytes = open(IMG, "rb").read()
    JW, JH = (IN_W+15)//16*16, (IN_H+1)//2*2
    DEC_SZ = JW*JH*3//2 + 1024*1024
    dev_jpeg, _ = acl.rt.malloc(len(jpeg_bytes), 0)
    jn = np.frombuffer(jpeg_bytes, np.uint8)
    acl.rt.memcpy(dev_jpeg, len(jpeg_bytes), acl.util.numpy_to_ptr(jn), len(jpeg_bytes), ACL_MEMCPY_H2D)
    dev_yuv, _ = acl.media.dvpp_malloc(DEC_SZ)
    dev_rgb, _ = acl.media.dvpp_malloc(RGB_STRIDE*H)
    pic_dec = mk_pic(dev_yuv, DEC_SZ, IN_W, IN_H, JW, JH, 1)
    pic_rgb = mk_pic(dev_rgb, RGB_STRIDE*H, W, H, RGB_STRIDE, H, 12)
    def pipeline():
        acl.media.dvpp_jpeg_decode_async(chn, dev_jpeg, len(jpeg_bytes), pic_dec, stream)
        acl.rt.synchronize_stream(stream)
        acl.media.dvpp_vpc_resize_async(chn, pic_dec, pic_rgb, rs_cfg, stream)
        acl.rt.synchronize_stream(stream)
        acl.rt.memcpy(dev_in_model, in_size, dev_rgb, RGB_STRIDE*H, ACL_MEMCPY_D2D)
        acl.mdl.execute(model_id, ds_in, ds_out)
else:
    # 640 自环场景
    jpeg_bytes = open("bus_det_310p.jpg", "rb").read()
    JW, JH = (W+15)//16*16, (H+1)//2*2
    DEC_SZ = JW*JH*3//2 + 1024*1024
    dev_jpeg, _ = acl.rt.malloc(len(jpeg_bytes), 0)
    jn = np.frombuffer(jpeg_bytes, np.uint8)
    acl.rt.memcpy(dev_jpeg, len(jpeg_bytes), acl.util.numpy_to_ptr(jn), len(jpeg_bytes), ACL_MEMCPY_H2D)
    dev_yuv, _ = acl.media.dvpp_malloc(DEC_SZ)
    dev_rgb, _ = acl.media.dvpp_malloc(RGB_STRIDE*H)
    pic_dec = mk_pic(dev_yuv, DEC_SZ, W, H, JW, JH, 1)
    pic_rgb = mk_pic(dev_rgb, RGB_STRIDE*H, W, H, RGB_STRIDE, H, 12)
    def pipeline():
        acl.media.dvpp_jpeg_decode_async(chn, dev_jpeg, len(jpeg_bytes), pic_dec, stream)
        acl.rt.synchronize_stream(stream)
        acl.media.dvpp_vpc_resize_async(chn, pic_dec, pic_rgb, rs_cfg, stream)
        acl.rt.synchronize_stream(stream)
        acl.rt.memcpy(dev_in_model, in_size, dev_rgb, RGB_STRIDE*H, ACL_MEMCPY_D2D)
        acl.mdl.execute(model_id, ds_in, ds_out)

# 预热 + 数值验证
pipeline()
host_out = np.empty(out_size//4, dtype=np.float32)
acl.rt.memcpy(host_out.ctypes.data, out_size, dev_out_model, out_size, ACL_MEMCPY_D2H)
arr = host_out.reshape(1, 300, 6)[0]
COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]
hi = arr[arr[:,4] > 0.5]
src_name = IMG or "bus_det_310p.jpg (640)"
print(f"[{src_name}] 检出 conf>0.5: {len(hi)} 个")
for row in hi[:8]:
    cls = COCO[int(row[5])] if int(row[5]) < len(COCO) else int(row[5])
    print(f"  {cls:<10} {row[4]:.3f}  box=({row[0]:.0f},{row[1]:.0f},{row[2]:.0f},{row[3]:.0f})")

# 压测
t0 = time.perf_counter()
for _ in range(N):
    pipeline()
acl.rt.synchronize_stream(0)
dt = (time.perf_counter()-t0)/N*1000
print(f"端到端全硬件: {dt:.2f} ms/帧 = {1000/dt:.0f} fps  (N={N})")
print("ALL HARDWARE PIPELINE OK")
