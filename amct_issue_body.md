### Describe the current behavior / 问题描述 (Mandatory / 必填)

在 **Atlas 300I Duo（Ascend 310P3）** 上使用 AMCT 开源版（amct_onnx 0.24.1，来自 `cann-amct-graph_9.2.0` 包）对 YOLO26n 做 INT8 PTQ，**全链路可跑通（量化校准 → ATC 编译 → OM 加载 → 真机执行），但 NPU 推理输出数值异常**：所有检测框 score < 0.02（fp16 基线为 0.89~0.92，5 个检出），mAP 为 0。

关键对照：**同一份 fake_quant ONNX 在 onnxruntime CPU 上数值完全正确**（6 检出，score 0.94，与 fp32 基线一致），说明校准产出的量化参数（scale/offset）本身无误，异常发生在 310P3 真机上的量化算子执行链路（AscendQuant → Conv(int8) → AscendDequant）。

注：AMCT 文档的产品支持矩阵仅列 Ascend 950 / Atlas A3 / Atlas A2，未列 310P。本报告将"文档未写"实测证实为"确实不可用"，并附完整复现与定位证据，供社区评估 310P 支持可行性与后续修复。

### Steps to reproduce the issue / 重现步骤 (Optional / 选填)

环境：

- 硬件：Atlas 300I Duo（Ascend 310P3），驱动/固件 25.5.1
- 量化（host 侧）：amct_onnx 0.24.1（cann-amct-graph 9.2.0），onnxruntime 1.16.0，自编译 `libamct_onnx_ops.so`
- ATC：CANN 8.2.RC1 工具链（`required_package_amct_acl_version="8.2"`，自带 `libamctacl.so`）
- 运行时：CANN 8.5.0 / 9.0.1 容器（均可加载执行）

复现：

```bash
# 1. 校准（159 张 MOT17/MOT20/VisDrone 混合图，activation_offset=false 对称量化）
amct_onnx calibration \
  --model yolo26n_pre_e2e_op16.onnx \
  --save_path ./yolo26n_int8 \
  --input_shape "images:1,3,640,640" \
  --data_dir ./calib_bin --data_types float32 \
  --calibration_config sym_config.pbtxt
# 产出 yolo26n_int8_deploy_model.onnx / fake_quant_model.onnx / quant.json

# 2. ATC 编译（8.2.RC1）
atc --model=yolo26n_int8_deploy_model.onnx --framework=5 \
    --output=yolo26n_int8_310p --soc_version=Ascend310P3 \
    --input_shape="images:1,3,640,640"     # 成功，OM 5.0MB

# 3. 真机推理：输出 1,300,6，score 全部 < 0.02
```

交叉验证（同一 fake_quant 模型，ORT CPU + amct 自定义算子库）：

```
CPU fake_quant: 6 检出, score 0.94（与 fp32 基线一致）  ← 量化参数正确
NPU OM:         0 检出, score max 0.02                  ← 真机执行异常
```

### Root cause / 根因分析

已排除的因素（均实测复现排除）：

1. **校准数据分布**：MOT17 单一场景与 MOT17+MOT20+VisDrone+bus 混合两套校准集，异常一致复现；
2. **activation_offset**：`activation_offset: true`（非对称，offset=-128/-127）与 `false`（对称）均复现；
3. **dst_type 属性**：deploy 图中 `AscendQuant.dst_type = 'INT8'`（字符串，正确）；
4. **量化参数本身**：CPU fake_quant 数值正确（见上）。

疑似方向（待官方确认）：

- 8.2.RC1 的 opp/TBE 中 `AscendQuant`（`dst_type` 仅定义 2=int8/6=int16/29=int4）、`Conv` INT8（W8A8）、`AscendDequant`（deq_scale 要求 6D NC1HWC0）这条链路在 310P3 上的实现或图优化（TransData 插入 88 处）存在问题；
- 高嫌疑点：per-channel dequant scale（uint64 打包的 fp32 scale+offset param）在 310P Conv INT8 输出（int32）上的应用环节。

**旁证——amct_pytorch classic 路线在同一环境下另有独立 bug**：其 deploy ONNX 经同一 ATC 编译成功，但 OM 在 8.0/8.5/9.0 三个运行时上加载均报 `SinkWeightData: sink data is invalid. src_size[27648], dest_size[23040]`（27648 = 432×64，对应首层 Conv 的 3×3×3×16 int8 权重 blob），疑似 8.2.RC1 ATC 对 AMCT 量化图的部分算子 weight sink 布局计算错误。两条独立量化路线各自命中不同的 8.2.RC1 编译/执行缺陷，说明 310P 上的 AMCT 量化部署链路整体未经充分验证。

### Related log / screenshot / 日志 / 截图

ATC 编译（成功，仅 2 条常规 W11001）：

```
ATC run success, welcome to the next use.
W11001: Op [/model.23/Mod] does not hit the high-priority operator information library...
```

真机推理（`--op_select_implmode=high_precision` 尝试过，OM 加载失败 ret=145002）：

```
model: yolo26n_int8_310p.om inputs=1 outputs=1
output shape: (1, 300, 6)
latency ms: avg=6.02 p50=6.02 p99=6.09     ← 执行正常
score col max = 0.017                       ← 数值全部异常
```

fp16 基线（同一输入、同一 310P3）：

```
5 检出: box=[39.1 235.4 185.9 536.0] score=0.920 cls=0
        box=[0.8 135.9 636.0 440.8] score=0.917 cls=5
        ...
```

复现材料可提供：`yolo26n_int8_deploy_model.onnx`（2.8MB）、`quant.json`、校准集清单、真机输出 npy。如需请告知上传渠道。
