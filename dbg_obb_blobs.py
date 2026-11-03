#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB execute 500002 诊断: desc 报 1 输入但底层 blobs 需要 2 组 —— 试探第二组 blob"""
import acl
import numpy as np

OM = "/work/yolo26n_obb_310p.om"
acl.init()
acl.rt.set_device(0)
ctx, r = acl.rt.create_context(0)
mid, r = acl.mdl.load_from_file(OM)
d = acl.mdl.create_desc()
acl.mdl.get_desc(d, mid)
n_in = acl.mdl.get_num_inputs(d)
n_out = acl.mdl.get_num_outputs(d)
print("desc num in/out:", n_in, n_out)
# 逐 index 试探 get_input_size, 看 index 1 是否存在
for i in range(n_in + 2):
    try:
        sz = acl.mdl.get_input_size_by_index(d, i)
        print(f"  in[{i}] size={sz}")
    except Exception as e:
        print(f"  in[{i}] err: {str(e)[:80]}")
        break
for i in range(n_out + 2):
    try:
        sz = acl.mdl.get_output_size_by_index(d, i)
        print(f"  out[{i}] size={sz}")
    except Exception as e:
        print(f"  out[{i}] err: {str(e)[:80]}")
        break
