"""YOLO26n INT8 静态量化（ORT QDQ，MOT17 真实图像校准）+ fp32/int8 精度 sanity
产物: yolo26n_int8_qdq.onnx
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                      CalibrationDataReader, CalibrationMethod)
import cv2
import glob

MODEL = 'yolo26n.onnx'
OUT = 'yolo26n_int8_qdq.onnx'
INPUT = 'images'


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]  # 1,3,640,640


class MotReader(CalibrationDataReader):
    """MOT17 train 图像做校准集（约 500 张，均匀采样）"""
    def __init__(self, n=500):
        imgs = sorted(glob.glob('/data2/gxw/mot_datasets/MOT17/train/*/img1/*.jpg'))
        step = max(1, len(imgs) // n)
        self.data = [preprocess(cv2.imread(p)) for p in imgs[::step][:n]]
        self.idx = 0
        print(f'calibration set: {len(self.data)} images from {len(imgs)} candidates')

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        d = self.data[self.idx]
        self.idx += 1
        return {INPUT: d}


if __name__ == '__main__':
    # 1. 量化（entropy 校准，per-channel 权重）
    quantize_static(
        MODEL, OUT,
        calibration_data_reader=MotReader(500),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Entropy,
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm'],
    )
    print('quantized ->', OUT)
    m = onnx.load(OUT)
    print('nodes:', len(m.graph.node))

    # 2. 精度 sanity：同一批 MOT 图，fp32 vs int8 输出对比
    sess32 = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
    sess8 = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
    imgs = sorted(glob.glob('/data2/gxw/mot_datasets/MOT17/train/*/img1/*.jpg'))[::37][:20]
    score_diffs, n_det32, n_det8 = [], 0, 0
    for p in imgs:
        x = preprocess(cv2.imread(p))
        o32 = sess32.run(None, {INPUT: x})[0]   # 1,300,6
        o8 = sess8.run(None, {INPUT: x})[0]
        d32 = o32[0][o32[0][:, 4] > 0.25]
        d8 = o8[0][o8[0][:, 4] > 0.25]
        n_det32 += len(d32)
        n_det8 += len(d8)
        if len(d32) and len(d8):
            # 按 (cx, cy) 最近邻粗配对，算 score 差
            for r in d32:
                dist = np.abs(d8[:, :2] - r[:2]).sum(axis=1)
                j = dist.argmin()
                if dist[j] < 30:
                    score_diffs.append(abs(d8[j, 4] - r[4]))
    print(f'images=20  det(fp32)={n_det32}  det(int8)={n_det8}')
    if score_diffs:
        print(f'score |diff|: mean={np.mean(score_diffs):.4f} max={np.max(score_diffs):.4f}')
