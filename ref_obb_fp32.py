#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""yolo26n-obb fp32 CPU 基线: 预处理 bus.jpg -> onnxruntime 推理 -> 保存输出 npy"""
import numpy as np
import cv2
import onnxruntime as ort

img = cv2.imread('/usr/local/lib/python3.10/dist-packages/ultralytics/assets/bus.jpg')
h, w = img.shape[:2]
r = min(1024 / h, 1024 / w)
nh, nw = int(h * r), int(w * r)
rs = cv2.resize(img, (nw, nh))
canvas = np.full((1024, 1024, 3), 114, dtype=np.uint8)
top, left = (1024 - nh) // 2, (1024 - nw) // 2
canvas[top:top + nh, left:left + nw] = rs
rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
x = rgb.transpose(2, 0, 1)[None]
x.tofile('input_obb_1x3x1024x1024.bin')

sess = ort.InferenceSession('yolo26n-obb.onnx', providers=['CPUExecutionProvider'])
out = sess.run(None, {'images': x})[0]
np.save('ref_obb_fp32.npy', out)
print('ref output:', out.shape)
