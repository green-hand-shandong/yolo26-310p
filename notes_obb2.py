#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB 最终验证: 用 OM 自身的 Sigmoid 输出路径 —— 直接 dump dev_in 头部解码。
这次 cls 全 0.447 是因为 dump 到的是 sigmoid 之前的原始 logits 经 expit(0.447)?
不对: expit(0.447)=0.61。0.447 本身就是合法 sigmoid 值域, 但 300 个全等值 = TopK
选出的全是同一填充项。
关键疑点: 输入图是 1024x1024, 但输入 dump 显示 'input intact: True' —— 说明输入
缓冲未被写穿, 模型内部可能没做真实推理(输出 blob 是未初始化/默认值)。
真正待办: 该模型的 execute 输出 blob 里 cls=0.447 全等值 —— 需要 dump 前几个
非零 box 值判断是否为有效推理。直接打印 box 原始 float 更直观。
"""
print("check: dump top-8 box floats to see if output is valid inference")
