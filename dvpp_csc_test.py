"""DVPP VPC: resize + CSC (YUV420SP -> RGB888) 一步完成, 输出直接给 NPU"""
import acl
import numpy as np
import time
acl.init(); acl.rt.set_device(0)
ctx,_ = acl.rt.create_context(0)

W,H = 640,640
IN_W,IN_H = 1280,720
# RGB888: 3 通道 packed = W*H*3, stride 对齐 16
OUT_SZ = W*H*3

chn = acl.media.dvpp_create_channel_desc()
ret = acl.media.dvpp_create_channel(chn)
print("vpc chn:", ret)

# 输入 YUV420SP
in_y_sz = IN_W*IN_H; in_uv = IN_W*IN_H//2
np_in = np.random.randint(0,255,in_y_sz+in_uv,dtype=np.uint8)
dev_in, r = acl.media.dvpp_malloc(in_y_sz+in_uv)
acl.rt.memcpy(dev_in, in_y_sz+in_uv, np_in.ctypes.data, in_y_sz+in_uv, 1)

pic_in = acl.media.dvpp_create_pic_desc()
acl.media.dvpp_set_pic_desc_data(pic_in, dev_in)
acl.media.dvpp_set_pic_desc_size(pic_in, in_y_sz+in_uv)
acl.media.dvpp_set_pic_desc_width(pic_in, IN_W)
acl.media.dvpp_set_pic_desc_height(pic_in, IN_H)
acl.media.dvpp_set_pic_desc_width_stride(pic_in, IN_W)
acl.media.dvpp_set_pic_desc_height_stride(pic_in, IN_H)
acl.media.dvpp_set_pic_desc_format(pic_in, 1)  # YUV420SP

# 输出 RGB888: 试 format=9 (常见 RGB888 编号), stride 对齐 16
for fmt_out in [9, 6, 12, 2]:
    stride_w = (W*3 + 15) // 16 * 16
    out_sz = stride_w * H
    dev_out, r = acl.media.dvpp_malloc(out_sz)
    pic_out = acl.media.dvpp_create_pic_desc()
    acl.media.dvpp_set_pic_desc_data(pic_out, dev_out)
    acl.media.dvpp_set_pic_desc_size(pic_out, out_sz)
    acl.media.dvpp_set_pic_desc_width(pic_out, W)
    acl.media.dvpp_set_pic_desc_height(pic_out, H)
    acl.media.dvpp_set_pic_desc_width_stride(pic_out, stride_w)
    acl.media.dvpp_set_pic_desc_height_stride(pic_out, H)
    acl.media.dvpp_set_pic_desc_format(pic_out, fmt_out)
    rs_cfg = acl.media.dvpp_create_resize_config()
    ret = acl.media.dvpp_vpc_resize_async(chn, pic_in, pic_out, rs_cfg, 0)
    acl.rt.synchronize_stream(0)
    host = np.empty(out_sz, dtype=np.uint8)
    acl.rt.memcpy(host.ctypes.data, out_sz, dev_out, out_sz, 2)
    nz = (host != 0).mean()
    print(f"out fmt={fmt_out}: ret={ret} 非零率={nz*100:.1f}% 首像素={host[:3]}")
    if ret == 0 and nz > 0.5:
        print(f"  → format {fmt_out} 有效! 输出 size={out_sz}B stride={stride_w}")
    acl.media.dvpp_destroy_pic_desc(pic_out)
    acl.media.dvpp_free(dev_out)
    acl.media.dvpp_destroy_resize_config(rs_cfg)
