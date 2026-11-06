"""4 进程 2 芯片：DP 扩展趋势"""
import sys, os, time
import multiprocessing as mp
sys.path.insert(0, '/work')
from dp_bench3 import worker

if __name__ == '__main__':
    N = 2000
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    ps = []
    for i, chip in enumerate([0, 1, 0, 1]):
        p = ctx.Process(target=worker, args=(chip, N, q, f'p{i}')); p.start(); ps.append(p)
    rs = [q.get() for _ in range(4)]
    for p in ps: p.join()
    total = sum(1000/r['ms'] for r in rs)
    print("4进程2芯片 各进程 ms/img:", [f"{r['ms']:.3f}" for r in rs])
    print(f"合计: {total:.0f} fps")
