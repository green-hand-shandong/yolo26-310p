# 310P 推理优化路线图（量化/剪枝/蒸馏——仅记录方案，未实施）

> 2026-11-03 记录。前提：fp16 基线已全部验证通过（检测 4.64ms / 批处理 364fps / 流水线 866fps / DVPP CSC 1.40ms）。

## 一、INT8 量化（推荐优先级 ★★★）

### 工具
昇腾 **AMCT**（Ascend Model Compression Toolkit），分 pytorch/tf/onnx 三种后端。
- pytorch 后端：`amct_pytorch`，直接吃 `.pt` 权重
- onnx 后端：`amct_onnx`，直接吃 `.onnx`（我们已有现成的）

### 方案
1. **PTQ（训练后量化）**：用 ~500 张校准图（bus.jpg 类似场景即可），无需重训
   ```bash
   amct_onnx calibration --model=yolo26n_pre_e2e.onnx --calibration_data=./calib/ --output=yolo26n_int8.onnx
   atc --model=yolo26n_int8.onnx --soc_version=Ascend310P3 ...
   ```
2. **QAT（量化感知训练）**：精度损失更小但需要重训 + 标注数据，成本高

### 预期收益（310P）
- INT8 算力是 fp16 的 **2 倍**：推理 4.64ms → 预计 **~2.5ms**
- 检测模型对小目标的 mAP 损失：通常 **<1%**（COCO 指标）
- 风险：TopK/Mod/Div 等后处理算子可能不支持 INT8，需要混合量化（backbone INT8 + head fp16）

### 实施前提
- AMCT 工具包（与 CANN 配套，昇腾官网下载，可能也需要账号）
- 校准数据集 500 张（可从客户场景视频抽帧）

## 二、结构化剪枝（推荐优先级 ★☆）

### 工具
- Torch-Pruning（开源，对 YOLOv8/11 支持好）
- Ultralytics 自带 `model.prune()`

### 方案
1. 对 yolo26s（9.5M 参数）做通道剪枝 → 接近 yolo26n 精度
2. 剪枝后 finetune 10~20 epoch 恢复精度
3. 重新导出 ONNX → ATC

### 预期收益
- yolo26s 剪枝 50% 通道：推理时间 ≈ yolo26n，但精度比 yolo26n 高
- **局限**：C2PSA 注意力模块的 MatMul 对通道数敏感，剪枝后可能需要结构适配

### 何时考虑
- 仅当 yolo26n 精度不够、yolo26s 太慢时才走这条路（当前 4.64ms 已经够快）

## 三、知识蒸馏（推荐优先级 ★☆）

### 工具
- 无官方 Ultralytics 蒸馏流程，需要手写
- 或者用 mmyolo/mmdetection 的蒸馏工具（但不支持 YOLO26 架构）

### 方案
1. Teacher: yolo26x 或 yolo26l（fp32，离线跑）
2. Student: yolo26n（现有模型）
3. 蒸馏损失：feature-level（C2PSA 输出对齐）+ logit-level（cls/box 分支）
4. 数据：客户场景标注数据（烟火、行为等）

### 预期收益
- yolo26n + yolo26x 蒸馏 → COCO mAP 提升 **~2-3 点**（41 → 43-44）
- 310P 推理时间不变（蒸馏只改权重，不改结构）

### 何时考虑
- 客户场景精度不达标时；需标注数据 + 训练资源（A100 或 910）

## 四、推理框架路线对比

| 路线 | 延迟 | 吞吐 | 实施难度 | 状态 |
|---|---|---|---|---|
| pyACL 同步 | 5.97ms | 168 fps | ✅ 已完成 | 基线 |
| pyACL 多线程流水线 | - | **866 fps** | ✅ 已完成 | 5.5× 基线 |
| 批处理 bs=4 | 2.75ms/img | **364 fps** | ✅ 已完成 | 2.2× 基线 |
| DVPP 硬件 CSC+resize | 1.40ms | **716 fps** | ✅ 已完成 | CPU 释放 10× |
| MindX SDK (mxVision) | - | 预计 >1000 fps | 🔴 需昇腾账号 | 阻塞 |
| INT8 (AMCT) | 预计 ~2.5ms | 预计 >500 fps | 🟡 需 AMCT 工具 | 方案已记录 |

## 五、结论：当前瓶颈与建议

当前 310P 单芯片的理论极限是：
- **NPU 纯推理**：~200 fps（fp16 bs=1）
- **DVPP 硬件预处理**：~700+ fps
- **多线程流水线**：~866 fps（瓶颈在 CPU YUV→RGB，用 DVPP CSC 后可释放）

**当前 866 img/s 已经是 pyACL + Python 的实际上限**，瓶颈在 Python 层调用开销。
要进一步突破有两条路：
1. **C++ 重写推理循环**（MindX SDK 或纯 ACL C++），预计 >1500 fps
2. **INT8 量化**，预计推理时间减半

MindX SDK 阻塞在下载渠道（需要昇腾社区账号），等账号到货后可继续。
