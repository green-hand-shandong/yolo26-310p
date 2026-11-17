### Describe the current behavior / 问题描述 (Mandatory / 必填)

在 **Atlas 300I Duo（Ascend 310P3）+ CANN 8.2.RC1** 上，`AveragePool` 算子（kernel 2x2、stride 1、count_include_pad=1，ONNX 导出）**执行结果数值错误**：含该算子的 YOLOv9t/v9c 检测模型在 NPU 上类别输出全错（v9t 仅错误检出 cls=35 'watch'、v9c 零检出），而**同一 ONNX 模型在 onnxruntime CPU 上完全正确**（50 检出、conf 0.93+）。

将该算子替换为数学等效的 depthwise Conv（固定权重 1/k²）后重新编译部署，**v9t/v9c 数值全部恢复正常**（50 检出、conf 0.928/0.959、person/bus 类别正确、延迟 7.37/12.11ms）。由此可锁定错误来源是 AveragePool 算子本身。

### Steps to reproduce the issue / 重现步骤 (Optional / 选填)

环境：
- 硬件：Atlas 300I Duo（Ascend 310P3），驱动/固件 25.5.1
- ATC：CANN 8.2.RC1 工具链
- 运行时：CANN 8.5.0 / 9.0.1 容器（均复现）

```bash
# 1. 原始模型（含 5 个 AveragePool, SPPELAN 结构）
atc --model=yolov9t.onnx --framework=5 --output=yolov9t_310p \
    --soc_version=Ascend310P3 --input_shape="images:1,3,640,640"   # 编译成功
# 真机推理: cls 输出全错（仅 cls=35 'watch' 低分检出），CPU ORT 同模型 50 检出 conf 0.928

# 2. 修复对照: AveragePool -> depthwise Conv (w=0.25, k=2, s=1, group=cin)
atc --model=yolov9t_convpool.onnx --framework=5 --output=yolov9t_convpool_310p \
    --soc_version=Ascend310P3 --input_shape="images:1,3,640,640"
# 真机推理: 50 检出, max conf 0.928, cls 分布 person×40/bus×10 —— 与 CPU 完全一致
```

AveragePool 属性（5 处，SPPELAN 内）：
```json
{"ceil_mode": 0, "count_include_pad": 1, "kernel_shape": [2,2], "pads": [0,0,0,0], "strides": [1,1]}
```

对照实验排除了其它因素：
- fp32 精度模式（`--precision_mode=allow_fp32_to_fp16`）同样错误 → 排除 fp16 精度；
- 权重与图一致（CPU ORT 正确）→ 排除导出问题；
- 其余 8 个 YOLO 版本（不含 AveragePool）全部数值正确 → 算子唯一性。

### Root cause / 根因分析

疑点集中在 CANN 8.2.RC1 opp 的 AveragePool TBE 实现（`op_impl/ai_core/tbe/impl/` 下 pool 算子）在
310P3 上对该参数组合（k=2, s=1, count_include_pad=1）的计算语义错误。
同类参数组合（如 k=2 s=2）未测，不确定影响面是否更大。

### Related log / screenshot / 日志 / 截图

原始模型 NPU 输出（错误）：

```
dets(conf>0.25): 4, cls hist: [(35, 4)]   # 'watch' —— bus.jpg 中不存在
max conf: 0.304
```

convpool 替换后（正确）：

```
dets(conf>0.25): 50, max conf: 0.928
cls hist: [(0 person, 40), (5 bus, 10)]   # 与 CPU ORT 基线一致
```

复现材料可提供：`yolov9t.onnx`（原始）、`yolov9t_convpool.onnx`（对照）、
真机输出 npy、CPU ORT 基线。如需请告知上传渠道。
