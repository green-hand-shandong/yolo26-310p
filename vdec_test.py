#!/usr/bin/env python3
"""310P VDEC 视频流硬解码排查脚本（CANN 8.2.RC1）

结论（2026-11-04 排查）：
1. pyACL 8.2 的 vdec_send_frame 真实签名是 5 参数 (k,k,k,k,O) —— 通过反汇编
   PyArg_ParseTuple 格式串 "kkkkO" + LD_PRELOAD 钩子双重确认。文档写的 3 参
   (desc, stream_desc, frame_config) 会报 "args parse failed"，3 参 ctypes 直调
   C 层直接 segfault（证明确实是 5 参 ABI）。
2. 5 参调用通过 SWIG 层后，C 层仍返回 100000（参数无效）。已排除：
   参数组合 15+ 种、回调线程模型（独立线程+context）、DVPP channel 共存。
3. 判定：CANN 8.2.RC1 在 310P3 上的 VDEC 运行时组件不完整（hi_mpi_vdec 数据面
   依赖媒体初始化流程未打通）。归入 CANN 9.x 升级后待办。
4. 替代方案见 video_e2e_test.py：FFmpeg CPU 解码 + DVPP VPC 硬件处理链路，
   95fps@1080p 已可满足多数场景。

本脚本保留最初 3 参调用形态，用于 CANN 升级后回归验证。
"""
import acl
import numpy as np
import time
import threading
import queue
import sys

FORMAT_H265_MAIN = 0
FORMAT_H264_BASELINE = 1
FORMAT_H264_MAIN = 2
FORMAT_H264_HIGH = 3
PIXEL_FORMAT_NV12 = 1

cb_queue = queue.Queue()


def vdec_callback(desc, frame):
    """VDEC 回调线程上下文；把结果丢回主线程处理"""
    try:
        ret = acl.media.dvpp_get_pic_desc_ret_code(frame)
        cb_queue.put((frame, ret))
    except Exception as e:
        cb_queue.put((None, -999))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "h264"
    path = f"/tmp/test_1080p.{which}"
    en_type = FORMAT_H264_MAIN if which == "h264" else FORMAT_H265_MAIN

    acl.init(); acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)

    # VDEC 回调线程：必须自己 set_device + create_context
    def cb_thread():
        acl.rt.set_device(0)
        cb_ctx, _ = acl.rt.create_context(0)
        while True:
            time.sleep(0.1)

    t = threading.Thread(target=cb_thread, daemon=True)
    t.start()
    time.sleep(0.3)

    desc = acl.media.vdec_create_channel_desc()
    acl.media.vdec_set_channel_desc_channel_id(desc, 0)
    acl.media.vdec_set_channel_desc_entype(desc, en_type)
    acl.media.vdec_set_channel_desc_out_pic_format(desc, PIXEL_FORMAT_NV12)
    acl.media.vdec_set_channel_desc_out_pic_width(desc, 1920)
    acl.media.vdec_set_channel_desc_out_pic_height(desc, 1080)
    acl.media.vdec_set_channel_desc_callback(desc, vdec_callback)
    thread_id = threading.get_ident()
    acl.media.vdec_set_channel_desc_thread_id(desc, thread_id)

    r = acl.media.vdec_create_channel(desc)
    print(f"vdec_create_channel({which}): ret={r}")
    assert r == 0, f"create channel failed {r}"

    # 读码流并按帧切分（AnnexB start code 00 00 00 01）
    data = open(path, "rb").read()
    startcodes = []
    i = 0
    while True:
        i = data.find(b"\x00\x00\x00\x01", i)
        if i < 0:
            break
        startcodes.append(i)
        i += 4
    print(f"码流 {len(data)} bytes, {len(startcodes)} NAL units")

    # 发送 NAL（简化：每帧=连续两个 NAL 以 SPS/PPS 或 IDR 开头）
    # 先只发前 30 个 NAL 验证回调
    sent = 0
    for k in range(len(startcodes) - 1):
        s, e = startcodes[k], startcodes[k + 1]
        chunk = data[s:e]
        dev_buf, _ = acl.rt.malloc(len(chunk), 0)
        np_chunk = np.frombuffer(chunk, np.uint8)
        acl.rt.memcpy(dev_buf, len(chunk), acl.util.numpy_to_ptr(np_chunk), len(chunk), 1)
        fs = acl.media.dvpp_create_stream_desc()
        acl.media.dvpp_set_stream_desc_data(fs, dev_buf)
        acl.media.dvpp_set_stream_desc_size(fs, len(chunk))
        fc = acl.media.vdec_create_frame_config()
        r = acl.media.vdec_send_frame(desc, fs, fc)
        sent += 1
        if sent >= 30:
            break

    # 收回调
    got = 0
    t0 = time.time()
    while got < 5 and time.time() - t0 < 10:
        try:
            frame, ret = cb_queue.get(timeout=1)
            got += 1
            print(f"回调 #{got}: ret_code={ret}")
        except queue.Empty:
            continue
    print(f"收到 {got} 帧回调" if got else "⚠️ 无回调")
    print("VDEC TEST", "OK" if got else "FAIL")


if __name__ == "__main__":
    main()
