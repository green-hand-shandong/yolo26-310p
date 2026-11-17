"""YOLO 各版本 310P 适配批量导出脚本
YOLO26: e2e (nms=False) —— 已验证基线
YOLO11/v8/v5u: 常规头, nms=False 导出（输出 1,84,8400），310P 上跑主干+NMS 外挂
YOLOv10: v10Detect 自带 e2e（无 TopK 后处理，直接输出 1,300,6）
全部 opset 17, batch 1, imgsz 640
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import sys

MODELS = {
    'yolo11n': dict(nms=False),
    'yolov8n': dict(nms=False),
    'yolov5nu': dict(nms=False),
    'yolov10n': dict(nms=False),  # v10 的 e2e 由模型结构决定，nms 参数无效
}

from ultralytics import YOLO

for name, kw in MODELS.items():
    onnx_path = f'/data2/gxw/yolo26-310p-eval/{name}.onnx'
    if os.path.exists(onnx_path):
        print(f'{name}: onnx exists, skip')
        continue
    try:
        m = YOLO(f'/data2/gxw/yolo26-310p-eval/{name}.pt')
        path = m.export(format='onnx', imgsz=640, opset=17, batch=1, simplify=True, **kw)
        print(f'{name}: exported -> {path}')
    except Exception as e:
        print(f'{name}: EXPORT FAILED: {type(e).__name__}: {str(e)[:200]}')
