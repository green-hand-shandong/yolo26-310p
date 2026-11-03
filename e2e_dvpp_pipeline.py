#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端 CPU+NPU 流水线: DVPP 硬件 resize/YUV->NPU 推理 -> 后处理
模拟真实视频流: 1280x720 YUV 帧 → DVPP resize 640x640 → yolo26n 推理 → 检测输出
对比: 纯CPU预处理 vs DVPP硬件预处理
"""
import acl
import numpy as np
import time
import threading
import queue
import sys

ACL_MEM_MALLOC_HUGE_FIRST = 0
DEV_ID = 0
W, H = 640, 640
Y_SZ, UV_SZ = W*H, W*H//2

# 模拟输入: 1280x720 YUV 帧 (真实视频解码后的格式)
IN_W, IN_H = 1280, 720
IN_YUV_SZ = IN_W*IN_H + IN_W*IN_H//2


def check(ret, msg):
    if isinstance(ret, (tuple, list)):
        ret = ret[-1]
    if ret != 0:
        print(f"[FAIL] {msg} ret={ret}")
        sys.exit(1)


class EndToEndPipeline:
    def __init__(self, om_path, n_workers=2):
        ret = acl.init()
        if ret not in (0, 500000):
            print("acl.init:", ret)
        check(acl.rt.set_device(DEV_ID), "set_device")
        self.ctx_main, r = acl.rt.create_context(DEV_ID)
        check(r, "ctx")

        self.model_id, r = acl.mdl.load_from_file(om_path)
        check(r, "load om")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.model_id), "desc")
        self.in_size = acl.mdl.get_input_size_by_index(self.desc, 0)
        self.out_size = acl.mdl.get_output_size_by_index(self.desc, 0)

        # DVPP 通道 (在 feeder 线程中创建, 绑定线程上下文)
        self._dvpp_ready = threading.Event()

        # DVPP 输入帧 (1280x720 YUV420SP)
        self.in_y_sz = IN_W*IN_H
        self.in_uv_sz = IN_W*IN_H//2
        self.dev_frame, r = acl.media.dvpp_malloc(self.in_y_sz + self.in_uv_sz)
        check(r, "dvpp frame")
        self.pic_in = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(self.pic_in, self.dev_frame)
        acl.media.dvpp_set_pic_desc_size(self.pic_in, self.in_y_sz + self.in_uv_sz)
        acl.media.dvpp_set_pic_desc_width(self.pic_in, IN_W)
        acl.media.dvpp_set_pic_desc_height(self.pic_in, IN_H)
        acl.media.dvpp_set_pic_desc_width_stride(self.pic_in, IN_W)
        acl.media.dvpp_set_pic_desc_height_stride(self.pic_in, IN_H + (IN_H % 16))
        acl.media.dvpp_set_pic_desc_format(self.pic_in, 1)  # YUV420SP

        # DVPP 输出帧 (640x640 YUV420SP)
        self.dev_rst, r = acl.media.dvpp_malloc(Y_SZ + UV_SZ)
        check(r, "dvpp rst")
        self.pic_out = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(self.pic_out, self.dev_rst)
        acl.media.dvpp_set_pic_desc_size(self.pic_out, Y_SZ + UV_SZ)
        acl.media.dvpp_set_pic_desc_width(self.pic_out, W)
        acl.media.dvpp_set_pic_desc_height(self.pic_out, H)
        acl.media.dvpp_set_pic_desc_width_stride(self.pic_out, W)
        acl.media.dvpp_set_pic_desc_height_stride(self.pic_out, H)
        acl.media.dvpp_set_pic_desc_format(self.pic_out, 1)

        # NPU 推理输入 (float32 RGB/255, 640x640) —— YUV->RGB 转换在 CPU 做 (后处理线程)
        self.host_f32 = np.empty(self.in_size // 4, dtype=np.float32)
        self.dev_npu_in, r = acl.rt.malloc(self.in_size, ACL_MEM_MALLOC_HUGE_FIRST)
        check(r, "npu in")
        self.host_out = np.empty(self.out_size // 4, dtype=np.float32)
        self.dev_npu_out, r = acl.rt.malloc(self.out_size, ACL_MEM_MALLOC_HUGE_FIRST)
        check(r, "npu out")

        self.np_in_ds = acl.mdl.create_dataset()
        r1 = acl.create_data_buffer(self.dev_npu_in, self.in_size)
        acl.mdl.add_dataset_buffer(self.np_in_ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
        self.np_out_ds = acl.mdl.create_dataset()
        r2 = acl.create_data_buffer(self.dev_npu_out, self.out_size)
        acl.mdl.add_dataset_buffer(self.np_out_ds, r2[0] if isinstance(r2, (tuple, list)) else r2)

        self.q = queue.Queue(maxsize=8)
        self.n_workers = n_workers
        self.stop_evt = threading.Event()

    def dvpp_resize(self):
        """DVPP 硬件 resize: 1280x720 -> 640x640"""
        check(acl.media.dvpp_vpc_resize_async(self.chn, self.pic_in, self.pic_out, self.rs_cfg, 0), "resize")
        # 取回结果到 host (做 YUV->RGB/255)
        yuv640 = np.empty(Y_SZ + UV_SZ, dtype=np.uint8)
        acl.rt.memcpy(yuv640.ctypes.data, Y_SZ+UV_SZ, self.dev_rst, Y_SZ+UV_SZ, 2)
        return yuv640

    def yuv_to_rgb_f32(self, yuv):
        """YUV420SP -> RGB float32/255 (CPU 实现, 生产中用 DVPP CSC 或 libyuv 优化)"""
        y = yuv[:Y_SZ].reshape(H, W).astype(np.float32)
        uv = yuv[Y_SZ:].reshape(H//2, W).astype(np.float32)
        u = uv[:, 0::2].repeat(2, axis=0).repeat(2, axis=1) - 128
        v = uv[:, 1::2].repeat(2, axis=0).repeat(2, axis=1) - 128
        r = y + 1.402 * v
        g = y - 0.344 * u - 0.714 * v
        b = y + 1.772 * u
        rgb = np.stack([r, g, b], axis=0)[None]  # (1,3,H,W)
        return np.clip(rgb / 255.0, 0, 1)

    def infer_worker(self):
        check(acl.rt.set_device(DEV_ID), "w set_device")
        ctx_w, r = acl.rt.create_context(DEV_ID)
        check(r, "w ctx")
        while not self.stop_evt.is_set():
            try:
                rgb = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if rgb is None:
                break
            self.host_f32[:] = rgb.ravel()
            acl.rt.memcpy(self.dev_npu_in, self.in_size,
                          acl.util.numpy_to_ptr(self.host_f32), self.in_size, 1)
            acl.mdl.execute(self.model_id, self.np_in_ds, self.np_out_ds)
            acl.rt.memcpy(acl.util.numpy_to_ptr(self.host_out), self.out_size,
                          self.dev_npu_out, self.out_size, 2)
            _ = self.host_out.reshape(1, 300, 6)[..., 4] >= 0.25  # 后处理
        acl.rt.destroy_context(ctx_w)

    def run(self, n_frames):
        """DVPP resize(硬件) -> YUV转RGB(CPU) -> NPU 推理 串联"""
        # 模拟视频帧数据
        frame = np.random.randint(0, 255, self.in_y_sz + self.in_uv_sz, dtype=np.uint8)
        acl.rt.memcpy(self.dev_frame, self.in_y_sz + self.in_uv_sz,
                      frame.ctypes.data, self.in_y_sz + self.in_uv_sz, 1)

        workers = []
        for _ in range(self.n_workers):
            t = threading.Thread(target=self.infer_worker, daemon=True)
            workers.append(t); t.start()

        # DVPP resize 在主线程 (通道绑定上下文), NPU 推理在 worker
        import threading as _th
        def feeder():
            check(acl.rt.set_device(DEV_ID), "feeder set_dev")
            ctx_f, r = acl.rt.create_context(DEV_ID)
            check(r, "feeder ctx")
            self.chn = acl.media.dvpp_create_channel_desc()
            check(acl.media.dvpp_create_channel(self.chn), "dvpp chn")
            self.rs_cfg = acl.media.dvpp_create_resize_config()
            self._dvpp_ready.set()
            for fi in range(n_frames):
                yuv = self.dvpp_resize()
                if fi == 0:
                    print(f"  [feeder] first resize done, yuv sz={yuv.size}")
                rgb = self.yuv_to_rgb_f32(yuv)
                if fi == 0: print(f"  [feeder] first rgb done shape={rgb.shape}")
                self.q.put(rgb)
                if fi == 0: print(f"  [feeder] first put done")
            for _ in range(self.n_workers):
                self.q.put(None)

        t0 = time.perf_counter()
        t_feed = _th.Thread(target=feeder, daemon=True)
        t_feed.start()

        got = 0
        while got < n_frames:
            item = self.q.get(timeout=30)
            if item is None:
                break
            got += 1
        t_end = time.perf_counter()

        self.stop_evt.set()
        for t in workers:
            t.join(timeout=5)

        total = t_end - t0
        return n_frames / total

    def cleanup(self):
        acl.media.dvpp_destroy_pic_desc(self.pic_in)
        acl.media.dvpp_destroy_pic_desc(self.pic_out)
        acl.media.dvpp_destroy_resize_config(self.rs_cfg)
        acl.media.dvpp_free(self.dev_frame)
        acl.media.dvpp_free(self.dev_rst)
        acl.media.dvpp_destroy_channel(self.chn)


if __name__ == "__main__":
    OM = sys.argv[1] if len(sys.argv) > 1 else "/work/yolo26n_e2e_310p.om"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print("=== 端到端: DVPP 硬件resize + CPU YUV->RGB + NPU 推理 ===")
    for nw in [1, 2, 4]:
        p = EndToEndPipeline(OM, n_workers=nw)
        thr = p.run(N)
        print(f"workers={nw}: e2e throughput = {thr:.1f} fps ({1000/thr:.2f} ms/帧, 1280x720源)")
        p.cleanup()

    print("\n对比基线: 纯CPU resize+推理 (同步) ≈ 158 img/s (6.32ms/帧)")
    print("对比基线: 多线程流水线 (CPU resize) ≈ 866 img/s")
