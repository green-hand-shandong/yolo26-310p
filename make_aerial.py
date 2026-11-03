#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBB 航拍验证: 用 DOTA 风格的合成航拍图(有旋转目标)做最终闭环。
由于没有 DOTA 数据集(用户要求不全量下载), 构造一张 1024x1024 的合成图:
深色背景 + 几个旋转矩形亮块, OBB 模型可能对这类简单图形有响应。
如果响应仍低, 则说明需要真实航拍图, 将其记录为待办。"""
import numpy as np
import cv2

# 构造合成航拍图: 深绿草地背景 + 旋转矩形(车辆/建筑轮廓)
img = np.full((1024, 1024, 3), (60, 80, 50), dtype=np.uint8)  # 草地色
# 画几个旋转矩形模拟车辆/建筑
rects = [((300, 300), (80, 40), 30), ((600, 500), (120, 60), -45),
         ((750, 200), (60, 30), 15), ((200, 700), (90, 45), 60)]
for center, size, angle in rects:
    box = cv2.boxPoints((center, size, angle)).astype(np.int32)
    cv2.fillPoly(img, [box], (180, 170, 160))  # 亮灰色块

cv2.imwrite('/data2/gxw/yolo26-310p-eval/aerial_syn.jpg', img)
# 预处理
h, w = img.shape[:2]
r = min(1024/h, 1024/w); nh, nw = int(h*r), int(w*r)
rs = cv2.resize(img, (nw, nh))
canvas = np.full((1024,1024,3), 114, dtype=np.uint8)
top, left = (1024-nh)//2, (1024-nw)//2
canvas[top:top+nh, left:left+nw] = rs
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
rgb.transpose(2,0,1)[None].tofile('/data2/gxw/yolo26-310p-eval/input_obb_aerial.bin')
print('合成航拍图已生成: aerial_syn.jpg + input_obb_aerial.bin')
