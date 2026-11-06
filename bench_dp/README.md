# DP 双芯片基准脚本集

配合 `notes_parallel_architecture.md` 的实测脚本。在 cann:8.5.0 容器中运行
（`--runtime=ascend -e ASCEND_VISIBLE_DEVICES=<chips>`），按文件名对应笔记中的实验编号：

| 文件 | 实验 |
|---|---|
| `dp_bench.py` / `dp_bench2.py` | 单芯片 vs 双芯片基线（v2 加分桶稳态测量） |
| `dp_bench3.py` / `dp_bench4.py` | 同芯片双进程 vs 双芯片（排除硬件互联）；4 进程扩展 |
| `dp_phase.py` | 分相测量：异步提交 vs 同步等待（submit 占比 100% 的证据） |
| `dp_stream.py` | 多 stream 轮转（证明 stream 并行度无法突破驱动瓶颈） |
| `dp_topology.py` | 同卡双芯 vs 跨卡双芯 vs 最远跨卡（拓扑无差异的证据） |
| `dp_all8.py` | 8 芯片全开扩展极限（4.81x） |
| `cpu_only.py` | 纯 CPU 双进程基线（排除 OS 调度） |
| `dp_deep.py` | batch 摊薄测量（bs1 369us vs bs8 43us/img） |
| `acl_cpp.cpp` | C++ ACL 对照（证明绕过 pyACL 无效） |
| `patch_maxpool.py` | maxpool 反传补丁的早期草稿（最终版见 `maxpool_bwd_patch.py`） |
