#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyACL 多线程流水线引擎: 输入线程 -> NPU 推理线程(pool) -> 后处理线程
流水线收益验证: 与单线程同步推理对比吞吐

架构:
  [生产者线程]  预处理图像 -> input_queue (预填 N_DEPTH 份)
  [推理线程 xW] acl.mdl.execute -> output_queue
  [消费者线程]  后处理(解码) -> 统计

关键: ACL 的 device context 是线程绑定的 —— 推理线程必须各自 create_context,
或者主线程 create_context 后其他线程 set ctx。这里用"每 worker 一个 ctx"模式。
"""
import acl
import numpy as np
import time
import threading
import queue
import sys

ACL_MEM_MALLOC_HUGE_FIRST = 0
DEV_ID = 0


def check(ret, msg):
    if isinstance(ret, (tuple, list)):
        ret = ret[-1]
    if ret != 0:
        print(f"[FAIL] {msg} ret={ret}")
        sys.exit(1)


class PipelineEngine:
    """多线程流水线推理引擎"""

    def __init__(self, om_path, n_workers=2, queue_depth=8):
        self.om_path = om_path
        self.n_workers = n_workers
        self.qd = queue_depth
        ret = acl.init()
        if ret not in (0, 500000):
            print("acl.init:", ret)
        check(acl.rt.set_device(DEV_ID), "set_device")
        # 主 context (模型加载用)
        self.ctx_main, r = acl.rt.create_context(DEV_ID)
        check(r, "create main ctx")

        self.model_id, r = acl.mdl.load_from_file(om_path)
        check(r, "load om")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.model_id), "get desc")
        self.in_size = acl.mdl.get_input_size_by_index(self.desc, 0)
        self.out_size = acl.mdl.get_output_size_by_index(self.desc, 0)
        self.in_shape = [d for d in
            acl.mdl.get_input_dims(self.desc, 0)[0]['dims']]
        print(f"model loaded: in_size={self.in_size} out_size={self.out_size} shape={self.in_shape}")

        # 输入队列: 预处理完成的 host buffer
        self.q_in = queue.Queue(maxsize=self.qd)
        # 输出队列: (host_out, timestamp)
        self.q_out = queue.Queue(maxsize=self.qd * self.n_workers)
        self.stop_evt = threading.Event()

    def producer(self, inputs):
        """生产者: 把预处理好的输入喂进队列"""
        for x in inputs:
            if self.stop_evt.is_set():
                break
            self.q_in.put(x)
        # 发送结束哨兵
        for _ in range(self.n_workers):
            self.q_in.put(None)

    def infer_worker(self, wid):
        """推理 worker: 独立 context, 循环取输入 -> execute -> 放输出"""
        # 关键: 每个 worker 线程绑定独立 context
        check(acl.rt.set_device(DEV_ID), f"w{wid} set_device")
        ctx_w, r = acl.rt.create_context(DEV_ID)
        check(r, f"w{wid} create ctx")

        dev_in, r = acl.rt.malloc(self.in_size, ACL_MEM_MALLOC_HUGE_FIRST)
        check(r, f"w{wid} malloc in")
        dev_out, r = acl.rt.malloc(self.out_size, ACL_MEM_MALLOC_HUGE_FIRST)
        check(r, f"w{wid} malloc out")

        in_ds = acl.mdl.create_dataset()
        r1 = acl.create_data_buffer(dev_in, self.in_size)
        acl.mdl.add_dataset_buffer(in_ds, r1[0] if isinstance(r1, (tuple, list)) else r1)
        out_ds = acl.mdl.create_dataset()
        r2 = acl.create_data_buffer(dev_out, self.out_size)
        acl.mdl.add_dataset_buffer(out_ds, r2[0] if isinstance(r2, (tuple, list)) else r2)

        host_in_buf = np.empty(self.in_size // 4, dtype=np.float32)
        host_out = np.empty(self.out_size // 4, dtype=np.float32)

        while not self.stop_evt.is_set():
            try:
                item = self.q_in.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            host_in_buf[:] = item.ravel()
            check(acl.rt.memcpy(dev_in, self.in_size,
                  acl.util.numpy_to_ptr(host_in_buf), self.in_size, 1), "H2D")
            acl.mdl.execute(self.model_id, in_ds, out_ds)
            check(acl.rt.memcpy(acl.util.numpy_to_ptr(host_out), self.out_size,
                  dev_out, self.out_size, 2), "D2H")
            self.q_out.put((host_out.copy(), time.perf_counter()))

        acl.rt.destroy_context(ctx_w)

    def run(self, inputs, postprocess=None):
        """启动流水线, 返回 (吞吐, 所有推理耗时)"""
        t_start = time.perf_counter()
        n = len(inputs)

        t_prod = threading.Thread(target=self.producer, args=(inputs,), daemon=True)
        workers = []
        for i in range(self.n_workers):
            t = threading.Thread(target=self.infer_worker, args=(i,), daemon=True)
            workers.append(t)

        t_prod.start()
        for t in workers:
            t.start()

        # 消费者在主线程: 取结果 + 后处理
        latencies = []
        got = 0
        t_first = None
        while got < n:
            out, ts = self.q_out.get(timeout=30)
            if t_first is None:
                t_first = time.perf_counter()
            if postprocess:
                postprocess(out)
            got += 1

        t_end = time.perf_counter()
        self.stop_evt.set()
        for t in workers:
            t.join(timeout=5)

        total = t_end - t_start
        e2e_lat = (t_end - t_first) * 1000 if t_first else 0
        return n / total, e2e_lat


def cleanup(self=None):
    pass


if __name__ == "__main__":
    OM = sys.argv[1] if len(sys.argv) > 1 else "/work/yolo26n_e2e_310p.om"
    N_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    IN_BIN = sys.argv[3] if len(sys.argv) > 3 else "/work/input_1x3x640x640.bin"

    base_input = np.fromfile(IN_BIN, dtype=np.float32).reshape(1, 3, 640, 640)
    # 生成 N 份不同输入(模拟不同帧): 加一点随机噪声
    rng = np.random.default_rng(42)
    inputs = [np.clip(base_input + rng.normal(0, 0.02, base_input.shape).astype(np.float32), 0, 1)
              for _ in range(N_IMG)]

    def postprocess(out):
        # 模拟后处理: score 过滤 (轻量)
        _ = out.reshape(1, 300, 6)[..., 4] >= 0.25

    # 对比: 单线程同步 vs 流水线 (1/2/4 workers)
    print("\n=== 基线: 单线程同步推理 ===")
    eng0 = PipelineEngine(OM, n_workers=0)  # 0 = 不启动 worker (单独测)
    # 直接同步跑
    acl.rt.set_device(DEV_ID)
    ctx0, r = acl.rt.create_context(DEV_ID)
    mid0, r = acl.mdl.load_from_file(OM)
    d0 = acl.mdl.create_desc()
    acl.mdl.get_desc(d0, mid0)
    in_sz = acl.mdl.get_input_size_by_index(d0, 0)
    out_sz = acl.mdl.get_output_size_by_index(d0, 0)
    dev_i, _ = acl.rt.malloc(in_sz, 0)
    dev_o, _ = acl.rt.malloc(out_sz, 0)
    ds_i = acl.mdl.create_dataset()
    r1 = acl.create_data_buffer(dev_i, in_sz); acl.mdl.add_dataset_buffer(ds_i, r1[0] if isinstance(r1,(tuple,list)) else r1)
    ds_o = acl.mdl.create_dataset()
    r2 = acl.create_data_buffer(dev_o, out_sz); acl.mdl.add_dataset_buffer(ds_o, r2[0] if isinstance(r2,(tuple,list)) else r2)
    hbuf = np.empty(in_sz//4, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(N_IMG):
        hbuf[:] = inputs[i].ravel()
        acl.rt.memcpy(dev_i, in_sz, acl.util.numpy_to_ptr(hbuf), in_sz, 1)
        acl.mdl.execute(mid0, ds_i, ds_o)
    t_sync = (time.perf_counter() - t0)
    thr_sync = N_IMG / t_sync
    print(f"sync throughput: {thr_sync:.1f} img/s ({t_sync*1000/N_IMG:.2f} ms/img)")

    print("\n=== 流水线: 多线程 ===")
    for nw in [1, 2, 4]:
        eng = PipelineEngine(OM, n_workers=nw)
        thr, e2e = eng.run(inputs, postprocess=postprocess)
        print(f"workers={nw}: pipeline throughput={thr:.1f} img/s | e2e last-batch lat={e2e:.1f} ms")
        acl.mdl.unload(eng.model_id)
