"""DVPP 能力探测: VPC(resize/csc) + VDEC(视频解码) 接口在 pyACL 中的可用性"""
import acl
acl.init(); acl.rt.set_device(0)
ctx,_ = acl.rt.create_context(0)

# 1. 检查媒体内存接口 (DVPP 需要 VPC 内存)
print("=== pyACL DVPP 相关接口 ===")
api = [a for a in dir(acl.media) if not a.startswith('_')]
print("acl.media:", api[:20])

# 2. 检查 VDEC 能力
print("\n=== VDEC (视频解码) ===")
vdec_api = [a for a in api if 'vdec' in a.lower() or 'vdec' in a]
print("vdec:", vdec_api)

# 3. 尝试最小 VPC resize
print("\n=== VPC (图像处理) 最小验证 ===")
try:
    # 创建 YUV420SP NV12 图像 64x64
    W, H = 64, 64
    Y_SZ, UV_SZ = W*H, W*H//2
    np_in = np.zeros(Y_SZ + UV_SZ, dtype=np.uint8)
    np_in[:Y_SZ] = 128  # Y 平面
    np_in[Y_SZ:] = 128  # UV 平面
    pic_in = acl.media.create_pic_desc()
    print("create_pic_desc:", pic_in if pic_in else "ret only")
except Exception as e:
    print("VPC probe err:", str(e)[:150])

# 4. dvpp malloc 测试
try:
    dev_buf, r = acl.media.dvpp_malloc(64*64*3//2)
    print(f"dvpp_malloc(6KB): ret={r} ptr=ok")
    acl.media.dvpp_free(dev_buf)
    print("dvpp_free: ok")
except Exception as e:
    print("dvpp_malloc err:", str(e)[:150])
