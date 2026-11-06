"""纯 CPU 微基准：排除 NPU 驱动，看进程扩展本身"""
import time, multiprocessing as mp

def spin(n, q):
    x = 0
    t0 = time.perf_counter()
    for i in range(n):
        x += i * i
    q.put((time.perf_counter()-t0, x))

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    N = 5_000_000
    p = ctx.Process(target=spin, args=(N, q)); p.start(); single = q.get(); p.join()
    ps = [ctx.Process(target=spin, args=(N, q)) for _ in range(2)]
    [p.start() for p in ps]; d1 = q.get(); d2 = q.get(); [p.join() for p in ps]
    print(f"单进程 CPU: {single[0]:.2f}s | 双进程: {max(d1[0],d2[0]):.2f}s | 比: {single[0]/max(d1[0],d2[0]):.2f}x")
