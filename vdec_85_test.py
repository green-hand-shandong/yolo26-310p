"""CANN 8.5 容器 VDEC 重测：标准 sample 姿势（回调线程独立 context + 完整 stream_desc）"""
import threading, time, queue, os, sys
import numpy as np

result_q = queue.Queue()
stop_flag = False

def vdec_callback(desc, frame):
    """VDEC 回调：在回调线程上下文执行"""
    try:
        ret = acl.media.dvpp_get_pic_desc_ret_code(frame)
        result_q.put(('frame', ret))
    except Exception as e:
        result_q.put(('err', str(e)))

def callback_thread():
    """VDEC 消息泵线程：必须独立 set_device + create_context"""
    import acl
    ret = acl.init()
    acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)
    global thread_id
    thread_id = threading.get_ident()
    ready.set()
    while not stop_flag:
        time.sleep(0.01)

import acl
ready = threading.Event()
thread_id = None

t = threading.Thread(target=callback_thread, daemon=True)
t.start()
ready.wait(timeout=5)

acl.rt.set_device(0)
main_ctx, _ = acl.rt.create_context(0)

# channel desc
desc = acl.media.vdec_create_channel_desc()
acl.media.vdec_set_channel_desc_channel_id(desc, 0)
# enType: H264_MAIN=2? 官方枚举 H264_BASELINE=1, H264_MAIN=2, H264_HIGH=3? 试遍
EN_TYPES = [1, 2, 3]
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1

H264_PATH = '/work/test_1080p.h264'
data = open(H264_PATH, 'rb').read()
print(f"h264 file: {len(data)} bytes", flush=True)

# 切 NAL（AnnexB startcode 00 00 00 01）
nals = []
i = 0
while True:
    j = data.find(b'\x00\x00\x00\x01', i)
    if j < 0: break
    nals.append(j)
    i = j + 4
print(f"NAL units: {len(nals)}", flush=True)

success = False
for en_type in EN_TYPES:
    acl.media.vdec_set_channel_desc_entype(desc, en_type)
    acl.media.vdec_set_channel_desc_out_pic_format(desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    acl.media.vdec_set_channel_desc_out_pic_width(desc, 1920)
    acl.media.vdec_set_channel_desc_out_pic_height(desc, 1080)
    acl.media.vdec_set_channel_desc_callback(desc, vdec_callback)
    acl.media.vdec_set_channel_desc_thread_id(desc, thread_id)
    r = acl.media.vdec_create_channel(desc)
    print(f"enType={en_type}: create_channel ret={r}", flush=True)
    if r != 0:
        continue

    # 发前 10 个 NAL
    sent = 0
    for k in range(min(10, len(nals))):
        s = nals[k]
        e = nals[k+1] if k+1 < len(nals) else len(data)
        chunk = np.frombuffer(data[s:e], dtype=np.uint8)
        dev_buf, r_malloc = acl.rt.malloc(len(chunk), 0)
        acl.rt.memcpy(dev_buf, len(chunk), acl.util.numpy_to_ptr(chunk), len(chunk), 1)
        fs = acl.media.dvpp_create_stream_desc()
        acl.media.dvpp_set_stream_desc_data(fs, dev_buf)
        acl.media.dvpp_set_stream_desc_size(fs, len(chunk))
        try:
            r_send = acl.media.vdec_send_frame(desc, fs, None)
            print(f"  NAL{k} send: {r_send}", flush=True)
            sent += 1
        except TypeError as te:
            # 3 参失败，试 4 参（加 frame_config）
            fc = acl.media.vdec_create_frame_config()
            try:
                r_send = acl.media.vdec_send_frame(desc, fs, fc)
                print(f"  NAL{k} send(4args): {r_send}", flush=True)
                sent += 1
            except Exception as e2:
                print(f"  NAL{k} FAIL: {str(te)[:50]} / {str(e2)[:50]}", flush=True)
                break
        acl.media.dvpp_destroy_stream_desc(fs)
        acl.rt.free(dev_buf)

    # 等回调
    got = 0
    t0 = time.time()
    while got < 3 and time.time() - t0 < 8:
        try:
            kind, val = result_q.get(timeout=1)
            got += 1
            print(f"  callback: {kind}={val}", flush=True)
        except queue.Empty:
            pass
    print(f"enType={en_type}: sent={sent} callbacks={got}", flush=True)
    if got > 0:
        success = True
        break
    acl.media.vdec_destroy_channel(desc)
    desc = acl.media.vdec_create_channel_desc()
    acl.media.vdec_set_channel_desc_channel_id(desc, 0)

print("VDEC 8.5 RESULT:", "SUCCESS" if success else "FAIL", flush=True)
