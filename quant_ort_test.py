"""ONNX Runtime 官方量化器 PTQ 测试（不依赖 AMCT）
对 yolo26n-cls.onnx 做 INT8 静态量化，验证 QDQ 模型能否被 ATC 接受
"""
import onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import numpy as np
import glob

model_path = '/tmp/yolo26n-cls.onnx'
out_path = '/tmp/yolo26n_cls_int8.onnx'

# 校准数据读取器：用随机数据模拟（正式量化换真实图）
class RandReader(CalibrationDataReader):
    def __init__(self, n=100):
        self.data = [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(n)]
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.data):
            return None
        d = self.data[self.idx]
        self.idx += 1
        return {'images': d}

quantize_static(
    model_path,
    out_path,
    calibration_data_reader=RandReader(100),
    quant_format=QuantFormat.QDQ,       # QDQ 格式（ATC 支持的量化表示）
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
print('quantized ok:', out_path)
m = onnx.load(out_path)
print('nodes:', len(m.graph.node))
