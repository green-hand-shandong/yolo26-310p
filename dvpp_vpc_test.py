#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DVPP VPC 硬件 resize/csc 最小验证: JPEG 解码 -> YUV420SP -> VPC resize 到 640x640"""
import acl
import numpy as np
import time

acl.init(); acl.rt.set_device(0)
ctx, _ = acl.rt.create_context(0)

W, H = 640, 640
Y_SZ, UV_SZ = W*H, W*H//2

# ===== 2. 创建 VPC 通道 =====
chn_desc = acl.media.dvpp_create_channel_desc()
ret = acl.media.dvpp_create_channel(chn_desc)
print("vpc channel:", ret)

# ===== 3. 构造输入 YUV420SP (模拟一张 1280x720 图) =====
IN_W, IN_H = 1280, 720
in_y_sz, in_uv_sz = IN_W*IN_H, IN_W*IN_H//2
np_in = np.zeros(in_y_sz + in_uv_sz, dtype=np.uint8)
np_in[:in_y_sz] = np.random.randint(0, 255, in_y_sz, dtype=np.uint8)
np_in[in_y_sz:] = np.random.randint(0, 255, in_uv_sz, dtype=np.uint8)

dev_in, r = acl.media.dvpp_malloc(in_y_sz + in_uv_sz)
print("dvpp_malloc input:", r)
r = acl.rt.memcpy(dev_in, in_y_sz+in_uv_sz, np_in.ctypes.data, in_y_sz+in_uv_sz, 1)
print("memcpy H2D:", r)

# ===== 4. VPC resize 1280x720 -> 640x640 =====
pic_in = acl.media.dvpp_create_pic_desc()
acl.media.dvpp_set_pic_desc_data(pic_in, dev_in)
acl.media.dvpp_set_pic_desc_size(pic_in, in_y_sz + in_uv_sz)
acl.media.dvpp_set_pic_desc_width(pic_in, IN_W)
acl.media.dvpp_set_pic_desc_height(pic_in, IN_H)
acl.media.dvpp_set_pic_desc_width_stride(pic_in, IN_W)
acl.media.dvpp_set_pic_desc_height_stride(pic_in, IN_H)
acl.media.dvpp_set_pic_desc_format(pic_in, 1)  # YUV420SP = 1

dev_out, r = acl.media.dvpp_malloc(Y_SZ + UV_SZ)
pic_out = acl.media.dvpp_create_pic_desc()
acl.media.dvpp_set_pic_desc_data(pic_out, dev_out)
acl.media.dvpp_set_pic_desc_size(pic_out, Y_SZ + UV_SZ)
acl.media.dvpp_set_pic_desc_width(pic_out, W)
acl.media.dvpp_set_pic_desc_height(pic_out, H)
acl.media.dvpp_set_pic_desc_width_stride(pic_out, W)
acl.media.dvpp_set_pic_desc_height_stride(pic_out, H)
acl.media.dvpp_set_pic_desc_format(pic_out, 1)

rs_cfg = acl.media.dvpp_create_resize_config()
ret = acl.media.dvpp_vpc_resize_async(chn_desc, pic_in, pic_out, rs_cfg, 0)
print("vpc_resize_async:", ret)

# ===== 5. 取回结果验证 =====
host_out = np.empty(Y_SZ + UV_SZ, dtype=np.uint8)
acl.rt.memcpy(host_out.ctypes.data, Y_SZ+UV_SZ, dev_out, Y_SZ+UV_SZ, 2)
y_plane = host_out[:Y_SZ].reshape(H, W)
print(f"输出 Y 平面: shape={y_plane.shape} mean={y_plane.mean():.1f} std={y_plane.std():.1f} (输入 std={np_in[:in_y_sz].std():.1f})")

# ===== 6. 性能: 100 次 VPC resize =====
N = 100
t0 = time.perf_counter()
for _ in range(N):
    acl.media.dvpp_vpc_resize_async(chn_desc, pic_in, pic_out, rs_cfg, 0)
    acl.rt.synchronize_stream(0)
dt = time.perf_counter() - t0
print(f"VPC resize 1280x720->640x640: {dt*1000/N:.2f} ms/帧 ({N/dt:.0f} fps)")

# 清理
acl.media.dvpp_destroy_resize_config(rs_cfg)
acl.media.dvpp_destroy_pic_desc(pic_in)
acl.media.dvpp_destroy_pic_desc(pic_out)
acl.media.dvpp_free(dev_in); acl.media.dvpp_free(dev_out)
acl.media.dvpp_destroy_channel(chn_desc)
print("DVPP PROBE DONE")
