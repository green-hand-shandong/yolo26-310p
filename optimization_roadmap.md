# 310P 推理优化路线图（量化/剪枝/蒸馏/多版本对比）

> 2026-11-03 记录方案；2026-11-09/10/11 实测更新。前提：fp16 基线已全部验证通过（检测 4.64ms / 216fps 单流）。

## 〇、YOLO 各版本 310P 适配对比（2026-11-11~17 实测，@640 bs1，同一芯片同一环境）

| 版本 | ATC 编译 | 延迟 (ms) | 吞吐 (fps) | bus.jpg 检出 | 输出格式 | 备注 |
|---|---|---|---|---|---|---|
| **YOLO26n** | ✅（9.0.1 编） | **4.64** | **215.6** | 5 ✓ | e2e (1,300,6) | 免 NMS，综合最优 |
| YOLOv8n | ✅ 一次过 | **4.13** | **242.3** | 5 ✓ | raw 84×8400 | 纯推理最快，需 host NMS |
| YOLOv5nu | ✅ 一次过 | 4.43 | 225.9 | 5 ✓ | raw | 需 host NMS |
| YOLO11n | ✅ 一次过 | 4.86 | 205.6 | 5 ✓ | raw | 需 host NMS；相比 v8 无优势 |
| YOLOv10n | ✅ 一次过 | 6.10 | 163.9 | 6 ✓ | e2e | 双头结构最慢 |
| YOLO12n | ✅ 一次过 | 7.84 | 127.6 | 5 ✓ | raw | A2C2f 注意力，慢 |
| YOLOv9t | ✅ 一次过 | 7.39 | 135.3 | ❌→**✅ 修复后** 50 检出 0.928 | raw | ⚠️ **AveragePool bug**，见下 |
| YOLOv9c | ✅ 一次过 | 12.11 | 82.6 | ❌→**✅ 修复后** 50 检出 0.959 | raw | 同上；模型大 |
| **YOLOv6n** | ✅（修 DOUBLE 常量后） | **3.12** | **320.5** | 52 ✓ | raw (1,8400,85) | **全场最快**；独立仓库 meituan/YOLOv6 |
| RT-DETR-L | ✅（GridSample 拆解后） | 289.6 | 3.5 | 5/5 匹配 CPU ✓ | e2e (1,300,6) | GridSample 拆解为 18×基础算子；**功能可行但拆解代价巨大** |
| RT-DETR-L 融合版 | ✅（batched GE+ReduceSum） | 282.5 | 3.5 | 5/5 ✓ | e2e | 融合版仅快 2.4%——**瓶颈是动态采样访存本身，非节点数** |

### 🐛 重要发现：8.2 opp 的 AveragePool 在 310P3 上数值错误
- **现象**：YOLOv9t/v9c 在 CPU ORT 上完全正常（50 检出、conf 0.93+），但 NPU 上 cls 端全错
  （v9t 只检出 cls=35 'watch'、v9c 零检出），box 端部分正常。
- **定位**：v9 独有 AveragePool×5（SPPELAN 结构），其余已测版本无此算子。
  fp32 精度模式同样错误 → 排除精度问题，是算子语义 bug。
- **修复**：把 AveragePool 替换为等效 depthwise Conv（固定权重 1/k²，`convpool` 替换）后
  v9t/v9c 数值全部恢复正常（50 检出、person/bus 正确）。
  脚本内嵌在会话记录；产物 `yolov9t_convpool_310p.om` / `yolov9c_convpool_310p.om`。
- **影响面**：所有使用 AveragePool 的模型（v9 系列）。已测其它版本（26/11/v8/v5/10/12）无此算子，不受影响。
- 可提 issue 到 cann 仓库（附 v9t ONNX + convpool 对照复现）。

### 结论（版本选择，310P）
1. **默认 YOLO26n**：e2e 免 NMS + 4.64ms，与 DVPP 流水线配合最优
2. **吞吐优先 + 自带 NMS**：YOLOv6n（**3.12ms / 320fps 全场最快**）或 YOLOv8n（4.13ms）
3. v12 慢（7.84ms）、v10（6.10ms）、v9 修复后可用但慢（7.4~12.1ms）——均不如 26/v6/v8
4. RT-DETR：GridSample 拆解后功能可行（5/5 检出匹配），但 289ms 不实用。融合版（单次 batched GatherElements + ReduceSum，节点数减 60%）也只到 282.5ms——**瓶颈是 18 个动态采样点在 AI Core 上的访存模式本身**，图层面无解；需要 CANN 原生 GridSample 算子或 CPU/NPU 异构（decoder 段回退 CPU）才实用
5. **检测模型已全覆盖**：26/11/12/v5/v6/v8/v9t/v9c/v10/RT-DETR 共 10 类
   —— 10 类全部编译通过、9 类数值正确、RT-DETR 可用但慢
6. 适配资产：`export_all_versions.py`、`gridsample_decomp.py`（GridSample 图拆解器，可复用于其它含 GridSample 的模型）、
   `test_om.py`（支持 84×8400 / 8400×85 双格式）、10 个版本的 OM 产物
7. 各版本适配踩坑：v6 需 nofuse 导出 + DOUBLE 常量修 FLOAT；v9 需 AveragePool→dwConv 替换；
   RT-DETR 需 GridSample 拆解；26/11/12/v5/v8/v10 直出

---

## 一、INT8 量化（推荐优先级 ★★★）

### 2026-11-09 实测：应用侧工具链三条路全部排除，唯一正路是 AMCT

| 路线 | 实测结果 | 结论 |
|---|---|---|
| ORT QDQ INT8 → ATC | 量化成功（`quant_int8.py`，MOT17 500 张 entropy 校准，per-channel，Conv/MatMul）；CPU sanity 检出 234→215（-8%）、score 偏差 mean 0.115；**但 ATC 解析直接失败**（CANN 8.5 与 9.0.1 均报 "No operator plugin for QuantizeLinear/DequantizeLinear"） | ❌ ONNX parser 不认 Q/DQ，死路 |
| msmodelslim 8.2.0（CANN 9.0.1 容器自带） | API 全部面向 LLM：w8a8/w4a8 dynamic、quarot、attention/linear processor，无 Conv 网络入口 | ❌ LLM 专用，CV 模型不适用 |
| ATC `--enable_compress_weight=true`（W8A16） | 编译通过，但 OM 体积 6.60MB → 6.60MB 零收益（yolo26n 权重太小，且 Cube 收益本就在 fp16） | ❌ 无收益 |
| fp8 | README 已实测：310P3 无 fp8 Cube 单元 | ❌ 硬件不支持 |

### 剩余唯一路线：AMCT —— ✅ 已开源，2026-11-09 全链路实测完成

**AMCT 已开源在 `gitcode.com/cann/amct`**（版本 9.2.0，Apache 2.0，已克隆到 `/data2/gxw/amct`）。
当日完成 YOLO26n INT8 PTQ 全链路实测，结果如下：

#### 实测打通的链路（amct_onnx 路线，推荐）
```
yolo26n_pre_e2e.onnx (opset 17→16 降级)
  → amct_onnx calibration（MOT17+MOT20+VisDrone 159 张校准，pbtxt 配置 activation_offset=false）
  → yolo26n_int8_deploy_model.onnx + fake_quant_model.onnx + quant.json
  → atc 8.2.RC1（gs310p 工具链，ccec 需在 PATH，ASCEND_OPP_PATH 指向 8.2 opp）
  → yolo26n_int8_310p.om（5.0MB，加载 ret=0 ✅，真机执行 ✅，6.0ms/帧）
```
- 量化脚本：`amct_onnx` CLI（whl 来自 `cann-amct-graph` 包，`install_graph.sh` 下载渠道 OBS 直链无需账号）
- amct_onnx 自定义算子库构建：需 onnxruntime 1.16 + ORT 头文件（github raw 被墙，用 gh-proxy.com 镜像下载）

#### 实测结论（310P3 真机）
| 项 | 结果 |
|---|---|
| ATC 编译 | ✅ 成功（需 8.2.RC1 ATC + ccec 在 PATH + PYTHONPATH 含 tbe） |
| OM 加载/执行 | ✅ 8.5/9.0 运行时均可加载执行 |
| 延迟 | **6.0ms**（fp16 基线 4.64ms——含 host 侧量化算子开销，未达 2× 预期） |
| 精度 | ❌ **NPU 输出 score 全部 <0.02，检出为 0**（fp16 基线 5 检出） |
| CPU 侧交叉验证 | ✅ 同一 fake_quant 模型在 ORT CPU 上检出 6/5 正常（score 0.94）——**量化参数本身正确** |

#### 精度异常定位（已排除项）
- ❌ 校准数据分布（MOT17 / MOT17+MOT20+VisDrone+bus 混合两套均复现）
- ❌ activation_offset（对称 offset=false 与非对称均复现）
- ❌ dst_type 属性（deploy 图为字符串 'INT8'，正确）
- ❌ 量化参数本身（CPU fake_quant 数值正确）
- 定性：**8.2.RC1 工具链在 310P3 上的 AscendQuant/Conv-int8/AscendDequant 链路 bug**
  （pytorch classic 路线另有 OM 加载 sink size 错误 src=27648/dst=23040，同为 8.2.RC1 问题）
- 附带发现：8.2.RC1 opp 缺 `libops_host_cpu.so`，需从 `compiler/lib64/` 手动复制到 `opp/built-in/op_impl/host_cpu/`

#### 后续路径
1. ~~向 gitcode cann/amct / cann Issues 提交 310P3 int8 数值 bug~~ ✅ 已提交 cann/amct#313，等官方回复
2. 等待 CANN 9.x 正式版（含新 amct_acl）后再回归——**当前 fp16 基线仍是最优可用配置**
3. 蒸馏路线不受影响（只改权重不改结构），可先行

### 绕过 AMCT 的自研量化路（2026-11-11 探索，可行性确认但撞同一堵墙）
- **关键发现**：AMCT deploy ONNX 的 AscendQuant/AscendDequant 用**默认域（''）**，8.2 ATC 的 ONNX parser 原生支持
  （opp plugin 有 84 处 AscendQuant/51 处 AscendDequant 注册）；之前手搓失败是因为放进了 amct.customop 域。
- **已验证**：手搓默认域最小量化模型（AscendQuant→Conv(int8)→AscendDequant）能通过 8.2 ATC 编译
  + 8.5/9.0.1 运行时加载 + 真机执行（`/tmp/mini_default_domain.onnx` → mini_dd.om）。
  **即：不依赖 AMCT 包，自研量化器（torch 校准 + 手工改图）工程上是可行的**。
- **但**：最小模型真机数值也不对（输出 94~366 vs CPU 参考 -0.9~3.2，无负值、比例 ~115x），
  与 AMCT deploy OM 的异常模式一致（cls logits 异常、语义错乱）——
  **同一 8.2 opp 量化算子链 bug**。默认域属性协议还有隐藏细节（dst_type 转换、fakequant_precision_mode、
  amct 插件在图优化期做的额外图变换），完整逆向是数天级工程，且大概率仍撞算子执行 bug。
- **结论**：310P3 上 INT8 的阻塞点在 **CANN 8.2 opp 算子执行层**，不在工具链——绕过 AMCT 也绕不开。
  等 CANN 9.x+ 的 amct/算子修复后回归。

### 工具
昇腾 **AMCT**（Ascend Model Compression Toolkit），分 pytorch/tf/onnx 三种后端。
- pytorch 后端：`amct_pytorch`，直接吃 `.pt` 权重
- onnx 后端：`amct_onnx`，直接吃 `.onnx`（我们已有现成的）

### 方案
1. **PTQ（训练后量化）**：用 ~500 张校准图（MOT17 已就绪），无需重训
   ```bash
   amct_onnx calibration --model=yolo26n_pre_e2e.onnx --calibration_data=./calib/ --output=yolo26n_int8.onnx
   atc --model=yolo26n_int8.onnx --soc_version=Ascend310P3 ...
   ```
2. **QAT（量化感知训练）**：精度损失更小但需要重训 + 标注数据，成本高

### 预期收益（310P）
- INT8 算力是 fp16 的 **2 倍**：推理 4.64ms → 预计 **~2.5ms**
- 检测模型对小目标的 mAP 损失：通常 **<1%**（COCO 指标）
- 风险：TopK/Mod/Div 等后处理算子可能不支持 INT8，需要混合量化（backbone INT8 + head fp16）

## 四、中间档模型（s 与 n 之间）—— 2026-11-10 实测，剪枝手法与数据量双重受阻

### 目标与方案演进
1. **AMCT 剪枝（失败）**：`pruning` 模块的 CNN domain 只认"扁平 Conv→BN→Conv"结构，
   ultralytics 的包装模块（Conv/C3k2/Concat/Detect）识别为 0 targets（`prune_diagnose` 实测）。
   自定义 `YOLO26PruningDomain`（`yolo26_prune_domain.py`，用模块图 from 索引建消费链）可剪
   backbone 下采样 Conv 链（5 个 target，-10% 参数），但 C3k2 内部 chunk(2) split +
   concat 多段结构无法用 AMCT 原生 `apply_keep_indices` 对齐，覆盖不了参数大头。**放弃此路**。
2. **官方缩放体系 + 通道子采样（thinning，第一版剪枝手法）**：yolo26.yaml 加自定义 scale（width 0.375）
   生成 `yolo26s-mid`（5.65M），`thinning_init.py` 把 s 的 COCO 权重按通道**均匀采样**迁移
   （chunk 分段对齐 + DWConv + head 分支，79 张量全部迁移）。
3. **Torch-Pruning DepGraph（业界主流，工程受阻）**：`tp_prune_yolo26s.py`。
   关键坑：① .pt 权重 requires_grad=False，必须先打开（否则 DepGraph 追踪得 0 组）；
   ② DepGraph 靠 autograd 反向图追踪，YOLO26 e2e 头（TopK/Gather）+ 126 conv @640 反向图
   极其庞大——640 输入跑 70 分钟未完成，64×64 小输入也内存爆炸（>50GB RSS）。**成本过高，搁置**。

### 业界剪枝手法对比（本仓库结论）
| 手法 | 通道选择 | 先验保留 | 本仓库实测 |
|---|---|---|---|
| 均匀子采样（thinning） | 按位置，无重要性评估 | 差（跨通道协同被打断） | bus.jpg 零检出，蒸馏 60ep 只恢复到 0.206 |
| L1/L2 范数 (Li 2017) | 权重范数 topk | 中 | 未测（被 AMCT/TP 结构问题挡住） |
| 激活统计 (AMCT variance/reconstruct) | 激活方差 + 最小二乘补偿 | 好（reconstruct 有输出补偿） | 结构不适配 ultralytics |
| DepGraph (TP 1.6) | 组联合打分 | 好 | autograd 追踪内存爆炸 |
| 缩放重训（ultralytics n/s/m 的做法） | 不剪，按 scale 重训 | 完美（但需要全量数据） | 结构已生成，训练受数据量限制 |

### 实测结果（MOT17，bs8，310P NPU）
| 方案 | best mAP50 | 备注 |
|---|---|---|
| mid 随机初始化 + 蒸馏 40ep | 0.119 | 无先验，学不动 |
| **mid thinning + 蒸馏 60ep (lr 5e-4)** | **0.206** | 两次独立训练一致（0.2041/0.2058） |
| mid thinning + 直接微调 60ep（对照） | 0.205 | **与蒸馏同水平** |
| 参考: n(COCO 官方权重) 微调 15ep | 0.405 | 2.57M 但有完整 COCO 先验 |

### 结论（双根因）
1. **手法粗糙**：均匀子采样无重要性评估，是业界已知最差手法；正确手法（reconstruct/DepGraph）
   在 ultralytics 结构上有工具适配障碍（AMCT 不认结构，TP 追踪爆炸）。
2. **数据量不足**：即使换好手法，先验恢复仍需大规模重训——1650 张 MOT17 不够（蒸馏=微调实测证明）。

### 中间档路线的可行出口（按推荐序）
1. **业务数据预训练**：若客户场景数据 ≥5 万张，mid + 蒸馏重训可达标
2. **COCO 子集重预训练**：下载 COCO train2017（~19GB），mid 从 s 蒸馏重预训练 50ep 再微调——成本 ~1 天/芯片
3. **接受现状**：n(4.64ms) 满足需求则无需中间档
4. 资产已就绪：`yolo26s-mid.yaml`、`thinning_init.py`、`yolo26_prune_domain.py`、
   `tp_prune_yolo26s.py`、`distill_mid.py`/`distill_mid_v2.py`、`prune_out/yolo26s-mid-coco.pt`

### 预期收益
- mid（width 0.375）FLOPs 约 s 的 56%：310P 延迟预计 5.36ms → ~3.8ms（若先验恢复到位）
- 精度目标：介于 n/s 之间的业务 mAP

### 何时考虑
- n 精度不够、s 延迟超标时的折中档；前提是有足量业务数据支撑先验重建

## 三、知识蒸馏（推荐优先级 ★★↑）—— ✅ 2026-11-10 小规模验证通过

### 实测结论（yolo26s → yolo26n，MOT17，310P NPU，15 epoch）

| 方案 | mAP50 | mAP50-95 | 说明 |
|---|---|---|---|
| **蒸馏**（teacher=yolo26s, dis=6.0） | **0.4053** | **0.1708** | score-weighted L2 特征蒸馏 |
| baseline（直接微调） | 0.3266 | 0.1418 | 同超参同数据 |
| **增量** | **+7.9 点** | **+2.9 点** | 小数据上收益显著 |

> 均为各 epoch 最佳值（两方案最佳都出现在 ep1，lr0=0.001 对 15 epoch 短训偏大，正式训练调小后差距预计更明显）。

### 实测链路（全部可复用）
- **数据**：`mot17_to_yolo.py` 把 MOT17 SDP gt 转成 YOLO 格式（train 1650 / val 750，行人单类，vis>0.3 过滤）→ `/data2/gxw/datasets/mot17_yolo/`
- **训练脚本**：`distill_mot17.py`（`baseline` / `distill` 两种模式），用法 `model.train(..., distill_model='yolo26s.pt', dis=6.0)`
- **NPU 环境**：docker 镜像 `yolo26-distill-env:310p`（mindie 9.0.1 + torch_npu 2.9 + ultralytics 8.4.189 + polars，已固化）。
  ⚠️ 重建环境必须 `pip install --no-deps ultralytics`，否则依赖解析会动 torch/torch_npu
- **补丁**：`maxpool_bwd_patch.py` 新增 `get_device_properties/get_device_capability` 的 'cuda:0' 别名兼容（ultralytics `_clear_memory` 路径需要）
- **速度**：bs8@640 约 1.2 it/s，15 epoch ≈ 35 分钟/芯片；teacher 前向无显著拖累
- 已知 fallback：`torchvision::nms` 在 NPU 回退 CPU（仅影响 val 的 NMS，训练不受影响）

### 后续
1. 正式训练调参：lr0 降至 2e-4、epoch 50+、可试 dis 3~10 与 teacher=yolo26m（gh-proxy 可下载）
2. 换业务数据蒸馏（收益最大场景）；蒸馏与量化正交，未来 CANN 修复 INT8 后可叠加
3. 模型导出闭环：best.pt → ONNX → ATC → 310P OM（`test_om.py` 直接可用）

### 方案
1. Teacher: yolo26s（本机已有 `.pt`；若要更高精度用 yolo26l/x，需 GPU/910 或 CPU 慢速跑）
2. Student: yolo26n（现有模型）
3. 数据：客户场景标注数据（或先用 MOT17 验证流程）
4. NPU 训练：复用 `train_finetune.py` + `maxpool_bwd_patch.py` 补丁集，加 `distill_model` 参数

### 预期收益
- yolo26n + 更大 teacher 蒸馏 → COCO mAP 预计提升 **~2-3 点**
- 310P 推理时间不变（蒸馏只改权重，不改结构）——与量化正交，可叠加

### 何时考虑
- 客户场景精度不达标时；这是"不增加推理成本提精度"的唯一手段

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
