#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB execute 500002 根因: 底层 blobs.size 与 model data.size 不匹配。
错误日志: 'data size:1 from model and input blobs.size:2 are not equal'
=> CANN 8.5 的 OBB 模型常量权重在 device 上以第 2 组 blob 形式存在但 desc 只暴露 1 输入。
绕过: 用 ASCEND_OPP / cann 9.0.1? 或直接检查 ATC 是否可用 --op_select_implmode。
本脚本先验证 'blobs' 数量假设: 检查 dataset 应有几个 buffer —— 通过二次分配全部 index 的 buffer (size=0 的除外)。
若 in[1] size=0 是隐藏 blob, 加载后无法直接喂 —— 改为检查错误日志提示的 blobs.size:2。
结论路径: 直接用 acl.mdl.execute 前先确认 dataset blob 数。改用 '两阶段' —— 先 dump
model 的 dynamic info。
"""
import acl
import numpy as np

OM = "/work/yolo26n_obb_310p.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc()
acl.mdl.get_desc(d, mid)
print("num inputs:", acl.mdl.get_num_inputs(d))
print("num outputs:", acl.mdl.get_num_outputs(d))
# dynamic batch info?
has_dyn, r = None, None
try:
    ret = acl.mdl.get_dynamic_batch(d)
    print("dynamic batch:", ret)
except Exception as e:
    print("dyn batch err:", str(e)[:100])
# profiling: model 的 profile num?
try:
    print("input dims detail:", acl.mdl.get_input_dims(d, 0))
    print("output dims detail:", acl.mdl.get_output_dims(d, 0))
except Exception as e:
    print("dims err:", str(e)[:100])
