# YOLO26 × Atlas 300I Duo (Ascend 310P) 适配验证

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](./LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend_310P3-green)
![CANN](https://img.shields.io/badge/CANN-8.5%20%7C%209.0.1-orange)

Ultralytics [YOLO26](https://docs.ultralytics.com/models/yolo26/) 在华为 Atlas 300I Duo（Ascend 310P3）推理卡上的
**端到端部署验证**：算子分析 → ONNX 导出 → ATC 转换（CANN 8.5 / 9.0.1）→ pyACL 真机推理 → 功能/数值/性能实测。

> **TL;DR：YOLO26（含 C2PSA 注意力）可以在 310P 上全 NPU 推理，无需任何 CPU 算子回退。
> yolo26n 单流 4.64ms（216 img/s），单芯片批量吞吐 ~574 img/s。**

## 核心结论

1. **YOLO26 的"端到端无 NMS"设计天然适合 310P**：e2e 导出仅含 384 个基础算子节点
   （Conv/TopK/Gather/MatMul/Softmax 等），无 NonZero、无 NonMaxSuppression、无 DFL——
   这些正是 310P 上的传统部署痛点算子。
2. **NMS 嵌入导出（`nms=True`）在 310P ATC 上直接编译失败**，必须使用端到端导出
   （`nms=False`），而这恰好是 YOLO26 的默认形态。
3. 全部算子 native 支持，ATC 仅 2 条 W11001 性能提示（topk 后处理的 Mod/Div 取整），影响可忽略。

## 性能数据（310P3 单芯片，fp16，640×640，bus.jpg）

| 模型 | batch | 延迟 avg (ms) | 吞吐 (img/s) |
|---|---|---|---|
| yolo26n | 1 | 4.64 (p50 4.63 / p99 4.69) | 215.7（单流） |
| yolo26n | 4 | 7.06 | 566.9 |
| yolo26n | 8 | 13.93 | 574.3（单芯片吞吐上限区） |
| yolo26n | 1×4进程并发 | ~7.7/路 | 517.5 |
| yolo26s | 1 | 5.36 (p99 5.43) | 186.7（单流） |

## 功能验证（bus.jpg，conf≥0.25）
- **yolo26n**：检出 5 个目标 person×4（0.89~0.92）+ bus（0.92），与 onnxruntime fp32 基线 **5/5 全匹配**（IoU>0.5 同类），score 平均绝对差 0.0012；画框图 `bus_det_310p.jpg`
- **yolo26s**：检出 6 个目标 bus（0.96）+ person×5（0.32~0.96），多检出 1 个低分遮挡行人，符合更大模型预期

## 目录结构

```
.
├── README.md                  # 本文档（含完整踩坑记录）
├── LICENSE                    # AGPL-3.0（与 Ultralytics YOLO26 同源）
├── export_onnx.py             # 一键导出部署友好 ONNX（nms=False）
├── test_om.py                 # pyACL 推理/测速脚本（固定 batch / 动态 batch）
├── infer_test.py              # 最小化冒烟测试脚本
├── input_1x3x640x640.bin      # bus.jpg 预处理输入（float32 RGB CHW /255）
├── bus_det_310p.jpg           # 310P 推理画框结果
├── yolo26n_pre_e2e.onnx       # yolo26n 静态 bs1 端到端 ONNX
├── yolo26n_cann901_bs1/bs4/bs8.om   # CANN 9.0.1 ATC 产物（Ascend310P3, fp16）
├── yolo26n_cann901_dyn.om     # 动态 batch(1/2/4/8) OM
├── yolo26n_e2e_310p.om        # CANN 8.5.0 ATC 产物
├── yolo26s_cann901_bs1.om     # yolo26s bs1 OM
└── yolo26n.pt / yolo26s.pt    # Ultralytics 官方预训练权重（不入库，见 .gitignore）
```

预训练权重请从 Ultralytics 官方下载（`yolo26n.pt` 会随 `ultralytics` 包自动下载），或直接使用本仓库内已转换好的 OM 模型。

## 验证环境
- 硬件：Atlas 300I Duo（Ascend 310P3，8 卡 16 芯片），测试卡 davinci0
- 驱动/固件：25.5.1（ascendhal 7.35.23, V100R001C23SPC006）
- ATC 转换：CANN 9.0.1（`quay.io/ascend/mindie:3.1.0-cann9.0.1-...-310p`，ATC 为 host 侧不需要驱动）
- 运行时：CANN 8.5.0（`m.daocloud.io/docker.io/ascendai/cann:8.5.0-310p-ubuntu22.04-py3.11`）
- Docker：**必须用 `--runtime=ascend -e ASCEND_VISIBLE_DEVICES=<n>`**（本机默认 runtime 即 ascend，自动挂设备+驱动）；手动 `--device/-v` 挂载方式在新镜像上会报 507899
- ultralytics：8.4.189（本机已装，含 YOLO26）

## 重要发现 / 踩坑记录
1. **CANN≥9.0 运行时与宿主驱动 25.5.1 不兼容**：mindie 3.1.0 (CANN 9.0.1)、vllm-ascend 0.30rc2 (9.2.0)、nightly (9.1.0)、gpustack cann8.5-mindie2.3 的运行时均报 `halGetDeviceInfo drvRet=87` → 507899。转换不受影响（ATC 纯 host 侧），运行时用 CANN 8.5 容器（ubuntu 基底 ascendai/cann:8.5.0-310p 可用）。**长期需升级宿主驱动至配套版本**。
2. **NMS 嵌入导出（nms=True）不能用于 310P**：ATC 直接失败（TBE 崩溃，NonMaxSuppression/NonZero 路径不支持）。必须用端到端导出（`nms=False`，默认），这也是 YOLO26 的设计初衷。
3. **静态 ONNX 只能转 bs=1**：`yolo26n_pre_e2e.onnx`（静态 batch=1）转 bs4/bs8 失败（图内 Reshape/锚点按 bs=1 硬编码）。转多 batch 或动态 batch 必须用 `dynamic=True` 导出的 ONNX。
4. **动态 batch OM 转换成功（13MB, 1/2/4/8 档），但 CANN 9.0.1 pyACL 的 `set_dynamic_batch_size` 返回 100000（多组参数/dtype/顺序均失败，疑似该版本 pyACL quirk）**。多档位需求可用固定多 OM 替代，或后续用 C++ ACL/MindX SDK 验证。
5. 本机 davinci2/6/7 等芯片被残留进程占用（独占模式），新容器报 507899；davinci0 空闲可用。选卡前用 `fuser /dev/davinciX` 或直接试 ASCEND_VISIBLE_DEVICES。
6. 基础 cann 镜像缺 TBE 依赖，ATC 前需 `pip install numpy decorator attrs psutil protobuf cffi sympy scipy`。

## 算子清单（e2e ONNX，384 节点，opset 17）
Conv×102, Mul×90 (SiLU), Sigmoid×88, Concat×24, Add×21, Split×12, Reshape×12,
Transpose×5, MatMul×4 (C2PSA 注意力), Unsqueeze×4, MaxPool×3 (SPPF),
GatherElements×3, Softmax×2, Resize×2 (Upsample nearest), Slice×2,
TopK×2 (e2e 选择), Expand×2, Sub, ReduceMax, Flatten, Mod, Div, Cast

**全部为 310P ATC 原生支持的基础算子；C2PSA（MatMul+Softmax 注意力）全程在 NPU 上执行，无 CPU 回退。**
ATC 仅 2 条 W11001 提示（`/model.23/Mod`、`/model.23/Div_1` 未命中高优先级算子库，位于 topk 后处理取整，已含在上述延迟数据中，影响可忽略）。

## 待办
- [ ] COCO 全集 mAP 精度回归（pt vs 310P OM fp16）
- [ ] 多卡并行吞吐（16 芯片理论 ~9200 img/s）
- [ ] 视频流端到端（解码→AIPP/YUV→推理）链路压测
- [ ] CANN 9.x 运行时启用（需宿主驱动升级至配套版本）
- [ ] 如需动态 batch：用 C++ ACL 验证 set_dynamic_batch_size（pyACL 9.0.1 有 quirk）

## 复现步骤
```bash
# 1. 导出 ONNX（本机 ultralytics>=8.4，预训练权重自动下载）
python3 -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').export(format='onnx', imgsz=640, opset=17, batch=1, nms=False)
# 多 batch/动态: YOLO('yolo26n.pt').export(format='onnx', dynamic=True, nms=False)"
# 注意：不要用 nms=True（NMS 嵌入版 ATC 失败）

# 2. ATC 转换（host 侧，任意新镜像均可；基础镜像需先装 TBE 依赖）
docker run --rm -v $PWD:/work quay.io/ascend/mindie:3.1.0-cann9.0.1-torch_npu2.9.0.post4-310p-openeuler24.03_lts-py3.11 \
  bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  atc --model=/work/yolo26n_pre_e2e.onnx --framework=5 --output=/work/yolo26n_bs1 \
      --soc_version=Ascend310P3 --input_shape='images:1,3,640,640'"

# 3. 真机推理（必须用 ascend runtime）
docker run --rm --runtime=ascend -e ASCEND_VISIBLE_DEVICES=0 -v $PWD:/work \
  m.daocloud.io/docker.io/ascendai/cann:8.5.0-310p-ubuntu22.04-py3.11 \
  bash -c "pip install -q 'numpy<2' && source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  python3 /work/test_om.py /work/yolo26n_cann901_bs1.om /work/out.npy 1 100"
```

## 来源与致谢

- 模型：[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)（AGPL-3.0），本仓库的 ONNX/OM 模型由其官方预训练权重转换而来，遵循相同许可证
- 硬件/工具链：华为 Atlas 300I Duo（Ascend 310P3）、CANN 8.5/9.0.1（ATC、pyACL）
- 验证日期：2026-10-29（结构验证）、2026-11-02（预训练权重功能/性能验证）


## YOLO26 各任务 310P 支持矩阵（2026-11-03 实测）

| 任务 | 预训练权重 | ATC | 真机推理 | 数值验证 | 延迟(bs=1) | 备注 |
|---|---|---|---|---|---|---|
| 检测 detect | ✅ yolo26n/s | ✅ 一次过 | ✅ | ✅ fp32 4/4 匹配 | 4.64ms | 已开源 |
| 姿态 pose | ✅ yolo26n-pose | ✅ 一次过 | ✅ | ✅ 关键点偏差<0.1px | 4.17ms | RLE 头无新增算子 |
| 实例分割 seg | ✅ yolo26n-seg | ✅ 一次过 | ✅ | ✅ 掩码系数偏差≤0.002 | 5.42ms | 多输出缓冲修复 |
| 分类 cls | ✅ yolo26n-cls | ✅ 一次过 | ✅ | ✅ top5 一致 | 0.65ms | 最轻任务 |
| OBB 旋转框 | ✅ yolo26n-obb | ✅ | ✅ | ⚠️ e2e 头 box 异常；one2many 路径已通 | 9.77ms | DOTA 航拍样本待补数值闭环 |
| 语义分割 sem | ❌ 无官方权重 | — | — | — | — | 仅 YAML 结构 |
| 深度估计 depth | ❌ 无官方权重 | — | — | — | — | 仅 YAML 结构 |
| 跟踪 track | ✅ detect + ByteTrack | — | ✅ 数据流验证 | — | — | detect 输出 [N,6] 直接对接 |

### OBB e2e 头 box 异常的根因与替代路径
- **现象**：e2e 导出的 OM，box 列输出值域 0~1（应为像素量级 3~1021），score 全 0.447（letterbox 灰度 114/255 的比例）
- **根因**：CANN 8.5/9.0.1 的动态 TopK+GatherElements 图存在 blob 地址重叠 bug（输出 blob 与输入 blob 共享内存）
- **替代路径（已验证）**：导出 `nms=True`（one2many），NMS 内嵌生效，box 值域正确（0~1006），9.77ms
- **遗留**：DOTA 航拍样本（非街景）的数值闭环验证，留待下一轮
