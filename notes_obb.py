#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB 正解: 输出 blob 写入的是 'score 列全 0.447' = 网络原始 logits 未解码。
真相: 该 OM 的输出 /model.23/Concat_6 是 300x7 的 [box, cls, angle] 原始拼接,
模型内没有 Sigmoid/decode —— YOLO26 end2end=False 时输出是原始 head 张量,
需要后处理! 而 fix2 那次读到 0.8392 是因为当时 dump 的是 dev_in 头部 = 上一次
实验遗留的输入缓冲内容, 不可信。
正确路径: end2end=False 的 YOLO26 输出 = (1, 300, 7)? 不对 —— (300,7) 就是
one2one 头的原始输出 [cx,cy,w,h,cls,angle,?]。cls 全 0.447 说明 Sigmoid 未做。
下一步: dump dev_in 头部后搜索 sigmoid 化的 cls —— 若 cls∈[0,1] 才算对。
真正要做: 检查 OM 内是否含 Sigmoid 节点; 若无, 后处理需自行补 expit。
"""
print("check: dump output head to see raw vs sigmoid values")
