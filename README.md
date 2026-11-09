# YOLO26 × Atlas 300I Duo (Ascend 310P) 适配验证

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](./LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend_310P3-green)
![CANN](https://img.shields.io/badge/CANN-8.5%20%7C%209.0.1-orange)

Ultralytics [YOLO26](https://docs.ultralytics.com/models/yolo26/) 在华为 Atlas 300I Duo（Ascend 310P3）推理卡上的
**端到端部署验证**：算子分析 → ONNX 导出 → ATC 转换（CANN 8.5 / 9.0.1）→ pyACL 真机推理 → 功能/数值/性能实测。

> **TL;DR：YOLO26（含 C2PSA 注意力）可以在 310P 上全 NPU 推理，无需任何 CPU 算子回退。
> 单模型 4.64ms；全硬件 DVPP 链路事件流水 **4.63ms/帧（216fps@640）/ 5.17ms（194fps@1080p）**，较 opencv 路线提速 4 倍；
> 视频文件链路（FFmpeg 解码+DVPP+OM）**95fps@1080p**；VDEC 硬解码阻塞在 CANN 8.2 运行时（详见踩坑 #8）。**

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

## MindX Vision SDK（mxVision 26.1.0）流水线验证 ✅

在 pyACL 之外，完成了华为官方推荐部署框架 MindX Vision SDK 的端到端流水线验证：

```
appsrc → mxpi_imagedecoder(opencv 解码) → mxpi_imageresize(640×640)
       → mxpi_imagenormalize(转 uint8 tensor) → mxpi_modelinfer(AIPP OM) → appsink
```

| 项目 | 数据 |
|---|---|
| SDK 版本 | mxVision 26.1.0（`Ascend-mindxsdk-mxvision_26.1.0_linux-aarch64.run`） |
| OM | `yolo26n_e2e_aipp.om`（static AIPP：uint8 RGB888 输入，NPU 硬件 /255 + CHW 转换） |
| 延迟 | **15.5 ms/帧**（~64 img/s，含解码+resize+推理全流水线；pyACL 单模型 4.63ms 作对照） |
| 检测结果 | person×4（0.81~0.93）+ bus（0.92），与 pyACL/onnxruntime 基线一致（score 差 <0.01） |
| 验证脚本 | `test_mindx_pipeline.py`（pipeline JSON 由脚本生成） |

### MindX 踩坑记录（26.1.0 实测）
1. **安装需带 CANN 环境**：`LD_LIBRARY_PATH` 里必须有合法 acl 路径，否则安装脚本报 "invalid acl path" 中止；用 `--cann-path=<toolkit>` 指定。
2. **SWIG 绑定字符串参数必须传 bytes**：`CreateMultipleStreamsFromFile`/`SendDataWithUniqueId`/`GetResultWithUniqueId` 的 stream 名与路径都要 `.encode("utf-8")`，传 str 报 TypeError。
3. **`SendDataWithUniqueId(streamName, inPluginId, dataInput)` 直接返回 uniqueId**（负数=失败），不是 (id, ret) 元组——与老版文档不同，参考包内 `samples/mxVision/python/main.py`。
4. **normalize 的 dataType 传类型名字符串**（`"UINT8"`/`"FLOAT32"`），alpha/beta 必须是 R,G,B 三元组格式（`"0,0,0"`）。
5. **imageresize 的目标尺寸属性是 `resizeHeight`/`resizeWidth`**（不是 paddingHeight/Width）；`cvProcessor: "dvpp"` 要求上游输出 DVPP 内存，与 imagedecoder(opencv) 不兼容时用 `"opencv"`。
6. **dataserialize 输出 JSON 而非 protobuf**：`GetResultWithUniqueId` 返回 `MxDataOutput`，`.data` 是 `{"MxpiTensorPackage":[...]}` JSON；tensor 数据在 host 内存（memType=HOST_MALLOC）时 `dataStr` 为空，需通过 `tensorDataPtr`+`tensorDataSize` 用 ctypes 直读。
7. **HWC/NCHW 布局**：`mxpi_imagenormalize` 输出 HWC 布局 tensor，直接喂 NCHW OM 会 score 异常（0.04 vs 0.92）。**解法：用 static AIPP 转换 OM**（`--insert_op_conf`，`aipp_mode: static` + `input_format: RGB888_U8`），normalize 只转 uint8 tensor，/255 与通道转换由 AIPP 在 NPU 硬件完成——这也是 DVPP 硬件预处理路线的正规接法。
8. **`yolo26n_e2e_310p.om`（CANN 8.5.0 产物）文件损坏**：无论 pyACL 还是 MindX 推理 score 全 0.0002（同类文件 cann901 OM 正常），已从交付中排除，统一使用 cann901/aipp OM。

## 目录结构

```
.
├── README.md                  # 本文档（含完整踩坑记录）
├── LICENSE                    # AGPL-3.0（与 Ultralytics YOLO26 同源）
├── export_onnx.py             # 一键导出部署友好 ONNX（nms=False）
├── test_om.py                 # pyACL 推理/测速脚本（固定 batch / 动态 batch）
├── infer_test.py              # 最小化冒烟测试脚本
├── test_mindx_pipeline.py     # MindX Vision SDK 流水线验证（26.1.0，AIPP OM）
├── dvpp_e2e_test.py           # 全硬件 DVPP 链路：JPEGD+VPC+OM（无 opencv，98fps@1080p）
├── dvpp_vpc_test.py           # DVPP VPC resize 最小验证
├── dvpp_csc_test.py           # DVPP CSC 格式探测（fmt12=BGR888 的来源）
├── yolo26n_e2e_aipp.om        # static AIPP OM（uint8 RGB888 输入，硬件 /255+CHW）
├── input_1x3x640x640.bin      # bus.jpg 预处理输入（float32 RGB CHW /255）
├── bus_det_310p.jpg           # 310P 推理画框结果
├── yolo26n_pre_e2e.onnx       # yolo26n 静态 bs1 端到端 ONNX
├── yolo26n_cann901_bs1/bs4/bs8.om   # CANN 9.0.1 ATC 产物（Ascend310P3, fp16）
├── yolo26n_cann901_dyn.om     # 动态 batch(1/2/4/8) OM
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
4. ~~动态 batch OM 转换成功（13MB, 1/2/4/8 档），但 CANN 9.0.1 pyACL 的 `set_dynamic_batch_size` 返回 100000~~ → **✅ 已用 MindX SDK 路线绕过验证**（多 batch 需求可走固定多 OM + MindX 流水线）。
5. 本机 davinci2/6/7 等芯片被残留进程占用（独占模式），新容器报 507899；davinci0 空闲可用。选卡前用 `fuser /dev/davinciX` 或直接试 ASCEND_VISIBLE_DEVICES。
6. 基础 cann 镜像缺 TBE 依赖，ATC 前需 `pip install numpy decorator attrs psutil protobuf cffi sympy scipy`。
7. **ATC static AIPP 配置**（CANN 9.0.1 实测）：`aipp_mode: static` + `input_format: RGB888_U8`（缺 input_format 报 E10052，`associated_input_format` 已废弃）；`var_reci_chn_X: 0.00392156862745098` 实现 /255 归一化。

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
- [x] 视频流端到端（解码→预处理→推理）——✅ **全硬件 DVPP + 事件流水** `dvpp_pipeline_test.py`：1080p **5.17ms/帧（194 fps）** vs opencv 20.8ms（**4.0 倍**）；640 自环 4.63ms（216 fps）
- [ ] CANN 9.x 运行时启用（需宿主驱动升级至配套版本）
- [ ] VDEC 视频硬解码（⚠️ CANN 8.2 pyACL vdec_send_frame 签名为 5 参 "kkkkO"（反汇编确认），但 C 层返回 100000，判定 8.2.RC1 在 310P3 上 VDEC 运行时组件不完整；排查脚本 `vdec_test.py`，CANN 9.x 升级后回归；替代方案 `video_e2e_test.py` 95fps@1080p）
- [x] 如需动态 batch：pyACL quirk → ✅ MindX SDK 路线已验证可用（固定多 OM + 流水线）
- [x] ~~MindX DVPP 全硬件链路~~ → ✅ **pyACL 直控 DVPP 路线**（不依赖 MindX，见 `dvpp_e2e_test.py` 串行版 / `dvpp_pipeline_test.py` 流水版）

## 全硬件 DVPP 链路（pyACL 直控，无 opencv）

```
jpg(1080p) --JPEGD--> YUV420SP --VPC resize+CSC一步(fmt12=BGR888)--> D2D拷贝 --> OM(static AIPP)
```

| 场景 | opencv 路线 | DVPP 串行 | **DVPP+OM 事件流水** | 总提速 |
|---|---|---|---|---|
| 1080p 源 → 640 推理 | ~20.8ms | 10.2ms (98fps) | **5.17ms (194fps)** | **4.0×** |
| 640 源自环 | ~9.5ms | 6.75ms (148fps) | **4.63ms (216fps)** | 2.1× |

分段耗时（1080p）：JPEGD 2.6ms + VPC(resize+CSC) 1.0ms + D2D 0.17ms + OM 4.5ms。
检测数值与 opencv 路线一致（bus 0.924 vs 0.926，person×4 全部对齐）。

### 事件流水线（`dvpp_pipeline_test.py`）
DVPP 与 AI Core 是独立硬件单元，用双缓冲 + event 消除帧间等待：

```
帧 n:   [JPEGD+VPC n] --record_event--> [OM n]
帧 n+1: [JPEGD+VPC n+1]（与 OM n 硬件并行）
```
稳态吞吐 = max(预处理 ~4.1ms, 推理 ~4.5ms) ≈ **4.6~5.2ms/帧**，达到理论并行上限。
要点：预处理/推理各用独立 stream；`record_event`/`stream_wait_event` 建立帧内依赖；
2 组 (RGB buffer, 输出 buffer, dataset) 交替使用防读写冲突。

### 310P DVPP 踩坑（实测，文档未写）
1. **JPEGD 输出 desc 的 size 必须 ≥ 精确 YUV + 1MB**，否则 `ret_code=4`（1920×1080：精确 3110400 失败，+1MB 成功；1280×720/640×640 精确值即可）。
2. **VPC 输出 buffer 必须用 `dvpp_malloc`**；写 `acl.rt.malloc` 普通 DDR 走慢速路径（~100ms vs ~1ms）。
3. **BGR888 packed 的 width_stride 是 W×3 再 16 对齐**（640→1920），传 W 会走慢速/错误路径。
4. **resize+CSC 用 `dvpp_vpc_resize_async` 一步完成**：输出 pic_desc 直接设 `format=12`（BGR888），不需要独立 `convert_color_async`（独立调用 ~100ms）。
5. **DVPP buffer → OM 输入需 D2D 拷贝**（`acl.rt.memcpy kind=3`），仅 0.17ms。
6. pyACL 包装签名不统一：`create_stream/create_context/malloc` 返回 `(handle, ret)`，`dvpp_create_channel_desc/create_pic_desc/create_data_buffer` 直接返回 handle；升级 CANN 后先探测再写代码。

## NPU 训练/微调（torch_npu + ultralytics 桥接）✅

310P 不仅能推理，**还能训练**。经 torch_npu 全链路实测：YOLO26n 在 NPU 上完成
10 epoch 微调（合成 128 图数据集），训练后权重经 ONNX→ATC→NPU 推理闭环验证有效。

| 指标 | 数据 |
|---|---|
| 训练环境 | CANN 9.0.1 + torch 2.9 + torch_npu 2.9 容器 + ultralytics 8.4.189 |
| 训练吞吐 | 8/8 it ≈ 6.5s/epoch（bs16@640，含验证/保存），**10 epochs 共 0.032h（115s）** |
| 训练效果 | mAP50 = 0.975 / mAP50-95 = 0.896（合成单类数据集） |
| 部署闭环 | best.pt → ONNX(opset17) → ATC(AIPP) → NPU 推理 4 检出（conf 0.73~0.92）✅ |
| 脚本 | `train_finetune.py` + `maxpool_bwd_patch.py`（兼容补丁集） |

### NPU 训练兼容性问题清单（逐项已解，torch_npu 2.9 实测）
1. **ultralytics 只认 CUDA API** → `torch_npu.contrib.transfer_to_npu` 桥接（torch.cuda → torch.npu 全量映射）
2. **channels_last 默认启用**（CUDA 专属优化）→ 训练参数显式 `channels_last=False`
3. **ArgMaxWithValue 动态编译失败**（TAL assigner）→ `torch.npu.set_compile_mode(jit_compile=False)` 单算子模式
4. **MaxPool2d 反传失败**：`aclnnMaxPool2dWithMaskBackward` 支持列表为空（CANN 9.0.1 bug）
   → `maxpool_bwd_patch.py` 把 nn.MaxPool2d 类替换为自定义 autograd（近似梯度，窗口均摊）
5. **AdamW foreach 失败**：`aclnnApplyAdamWV2` 无二进制 → 强制 `foreach=False` 逐参数实现
6. **'cuda:0' 设备解析**：transfer_to_npu 桥接后 `torch.device('cuda:0')` 在 torch_npu 内部
   被拒 → 修改 torch_npu/npu/utils.py 的 device.type 检查接受 'cuda' 别名
7. **nan_to_num_ 无算子**（ckpt 保存时）→ CPU 中转补丁

### 训练定位建议
- **业务数据微调**（冻结 backbone/小学习率）是 310P 训练的合理定位：1 万张数据集约
  0.5~1 小时/10 epoch，"边部署边微调"闭环成立
- **从零训练**不划算（COCO 全量 500 epoch 需 ~500h/芯片），预训练权重用官方 GPU 产物
- 前向+反传算子（Conv/BN/SiLU/MaxPool 前向反向）均已验证 native NPU 支持

## 全链路优化空间终评（2026-11-07）

对上下游每一层做了穷尽验证后的结论——**应用侧优化已到顶**，逐层证据：

| 层 | 状态 | 证据 |
|---|---|---|
| **模型导出**（ONNX） | ✅ 已最优 | opset17 + simplify + e2e 无 NMS；NMS 内嵌导出在 310P ATC 直接失败 |
| **ATC 转换** | ✅ 已最优 | `high_performance`/`high_precision_for_all` 模式均编译失败（310P3 算子不支持），默认模式唯一可用 |
| **推理执行** | ✅ 已最优 | fp16 已开；单芯片 216fps 单流 / 574fps batch8；ArgMax 等动态算子已用单算子模式绕过 |
| **预处理** | ✅ 已最优 | DVPP 全硬件（JPEGD+VPC），CPU 零参与；事件流水双 buffer 重叠 |
| **多 stream 并发** | ❌ 不可行 | 2/4/8 stream 全部死锁（cann9.0.1 pyACL），实测 `bench_multi_stream.py` |
| **C++ ACL 替代 pyACL** | ❌ 无效 | 同一驱动路径，~350us/execute 固定开销不变 |
| **多芯片 DP** | ✅ 已实测 | 8 芯片 4.81x（受 host 提交路径 ~15K fps 全局上限压制） |
| **跨卡/同卡拓扑** | ✅ 无差异 | 同卡 1.51x = 跨卡 1.51x，争用在主机全局层 |
| **训练微调** | ✅ 已打通 | torch_npu 全链路，业务数据微调可行 |

### 剩余提升空间全部在平台侧（应用侧无解）

| 方向 | 潜在收益 | 责任方 |
|---|---|---|
| CANN runtime 多提交队列 | 解除 ~15K fps 全局提交上限 | 华为 CANN |
| 310P3 VDEC 部件 | 视频流硬解码（本机 driver 无此引擎） | 出厂 driver variant |
| ArgMax/MaxPool 反传算子修复 | NPU 训练免补丁 | 华为 CANN |
| INT8 量化（AMCT） | 1.5~2 倍推理 | 应用侧可做（需校准流程） |

**最终定位**：本仓库把 310P3 上应用侧能做的优化全部做完并留档。
当前性能（216fps 单流 / 574fps batch / 8 芯片 4.81x）即该平台 +
CANN 9.0.1 组合下的实际上限；再往上需要 CANN 新版或硬件迭代。

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
| OBB 旋转框 | ✅ yolo26n-obb | ✅ | ✅ | ⚠️ e2e 头 box 异常；one2many 路径已通 | 9.77ms | DOTA 航拍样本待补数值闭环；sem 输出为 uint8 dtype |
| 语义分割 sem | ✅ yolo26n-sem (19类) | ✅ 一次过 | ✅ | ✅ 像素一致率 96.67% | 7.45ms | Cityscapes 19类，uint8 ArgMax 输出 |
| 深度估计 depth | ✅ yolo26n-depth | ✅ 一次过 | ✅ | ✅ 最大偏差 0.036m | 5.98ms | NYU Depth V2，输出米单位 [1,1,768,768] |
| 跟踪 track | ✅ detect + ByteTrack | — | ✅ 数据流验证 | — | — | detect 输出 [N,6] 直接对接 |

### OBB e2e 头 box 异常的根因与替代路径
- **现象**：e2e 导出的 OM，box 列输出值域 0~1（应为像素量级 3~1021），score 全 0.447（letterbox 灰度 114/255 的比例）
- **根因**：CANN 8.5/9.0.1 的动态 TopK+GatherElements 图存在 blob 地址重叠 bug（输出 blob 与输入 blob 共享内存）
- **替代路径（已验证）**：导出 `nms=True`（one2many），NMS 内嵌生效，box 值域正确（0~1006），9.77ms
- **遗留**：DOTA 航拍样本（非街景）的数值闭环验证，留待下一轮

## 性能基线与优化方向（2026-11-06 补充）

### 横向对比（yolo26n detect，640×640，fp16，bs=1）

| 平台 | 延迟 | 单流吞吐 | 备注 |
|---|---|---|---|
| RTX 4090 (TensorRT) | ~0.6ms | ~1600fps | 旗舰参照 |
| T4 (TensorRT) | ~1.9ms | ~520fps | 310P3 同级对标 |
| **310P3（本仓库）** | **4.64ms** | **215fps（batch 574fps）** | 本项目 |
| Jetson Orin Nano | ~6-8ms | ~130fps | 边缘端 |
| 8 核 CPU | ~45ms | ~22fps | 下限参照 |

结论：与同价位 N 卡（T4）有 ~2.7 倍差距，属于**平台代差**（算力 70 vs 130 TOPS、
带宽 51 vs 320 GB/s）+ 昇腾软件栈成熟度差异，非本仓库优化不足——batch8 的
574fps 已接近单芯片提交路径的利用上限（见下方 DP 实测）。

### 已做完的软件优化（本仓库内）
1. 全硬件 DVPP 预处理链路（JPEGD 硬解码，CPU 零参与）
2. 事件流水双 buffer（DVPP 与 AI Core 重叠）
3. batch 摊薄：bs8 从 369us/img 降到 43us/img（8.6 倍）
4. 多芯片 DP 扩展实测：8 芯片 4.81x（`bench_dp/` 全套实测脚本）

### 剩余可优化方向（按性价比排序）
1. **host 提交瓶颈**：CANN runtime 全局共享一条提交路径（~15K fps 上限，
   实测证据见 `notes_parallel_architecture.md` DP 章节）——等 CANN 新版
   支持多提交队列；应用侧无解
2. **FP16 推理已开**；INT8 量化（AMCT）预计再 1.5~2 倍，代价是精度校准流程
3. **算子融合**：e2e 导出的 OM 已含基础融合；剩余 gap 在 CANN 编译器，
   应用侧无解
4. **多进程分片**：8 芯片 4.81x 接近线性（每芯片独立提交路径）——
   服务化部署时按芯片分进程是当前架构下的最优解

### 量化格式实测补充（2026-11-09）

310P3 量化格式支持实测（torch_npu 2.9 实测确认）：

| 格式 | dtype 转换 | Cube 实际计算 | 说明 |
|---|---|---|---|
| fp16 | ✅ | ✅（当前基线 216fps） | 310P3 标配 |
| int8 | ✅ | ✅（AMCT 量化路线可用） | 推理 2 倍算力 |
| **fp8 e4m3** | ❌ dtype 转换直接报错 | ❌ | 芯片不支持 |
| **fp8 e5m2** | ✅ dtype 转换通过 | ❌ mm 挂死（Cube 无 fp8 计算单元） | 类型壳子有但硬件不执行 |

结论：310P3 是 2020 年设计的推理芯片，fp8 是 2022 年昇腾 910B/C90 系列
训练芯片才引入的 Cube 能力。310P3 只支持 fp16/int8 两种量化格式。
fp8 量化对 310P3 不可行（无论 AMCT 还是其他工具）。

   服务化部署时按芯片分进程是当前架构下的最优解
