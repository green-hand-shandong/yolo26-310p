#!/usr/bin/env python3
"""MindX Vision SDK (mxVision 26.1.0) minimal pipeline test for YOLO26n e2e detect OM.

Pipeline: appsrc -> mxpi_imagedecoder -> mxpi_imageresize -> mxpi_modelinfer -> appsink

YOLO26 e2e head outputs final boxes (no NMS needed), so we skip
mxpi_objectpostprocessor and parse the MxpiTensorPackage output directly
via protobuf (26.1.0 returns MxpiTensorPackageList from appsink).

API notes (mxVision 26.1.0):
  - StreamManagerApi.CreateMultipleStreamsFromFile(json)
  - SendDataWithUniqueId(streamName, dataInput) -> uniqueId
  - GetResultWithUniqueId(streamName, uniqueId, timeoutMs)
    -> list of MxBufferAndMetadataOutput (mist_output)

Usage:  python3 test_mindx_pipeline.py [image.jpg]
"""
import json
import os
import sys
import time

from StreamManagerApi import StreamManagerApi, MxDataInput
from StreamManagerApi import MxBufferAndMetadataOutput
import MxpiDataType_pb2

MX_SDK_HOME = os.environ.get("MX_SDK_HOME", "/data2/gxw/yolo26-310p-eval/mxVision-26.1.0")
BASE = os.path.dirname(os.path.abspath(__file__))
OM_PATH = os.path.join(BASE, "yolo26n_e2e_aipp.om")  # AIPP: uint8 in, /255 + CHW on NPU
DEVICE_ID = 0
MODEL_W = MODEL_H = 640
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"]

PIPELINE_JSON = {
    "imagedecoder": {
        "stream_config": {"deviceId": str(DEVICE_ID)},
        "appsrc0": {"factory": "appsrc",
                    "props": {"blocksize": "409600"},
                    "next": "mxpi_imagedecoder0"},
        "mxpi_imagedecoder0": {"factory": "mxpi_imagedecoder",
                               "props": {"cvProcessor": "opencv"},
                               "next": "mxpi_imageresize0"},
        "mxpi_imageresize0": {"factory": "mxpi_imageresize",
                              "props": {"cvProcessor": "opencv",
                                        "resizeType": "Resizer_KeepAspectRatio_Fit",
                                        "resizeHeight": str(MODEL_H),
                                        "resizeWidth": str(MODEL_W)},
                              "next": "mxpi_imagenormalize0"},
        # AIPP OM takes 640x640x3 uint8 RGB888 directly:
        # normalize only converts image -> MxpiTensor (uint8), AIPP does /255+CHW on NPU
        "mxpi_imagenormalize0": {"factory": "mxpi_imagenormalize",
                                 "props": {"dataSource": "mxpi_imageresize0",
                                           "processType": "1",
                                           "dataType": "UINT8"},
                                 "next": "mxpi_modelinfer0"},
        "mxpi_modelinfer0": {"factory": "mxpi_modelinfer",
                             "props": {"dataSource": "mxpi_imagenormalize0",
                                       "modelPath": OM_PATH,
                                       "outputDeviceId": "-1"},
                             "next": "mxpi_dataserialize0"},
        "mxpi_dataserialize0": {"factory": "mxpi_dataserialize",
                                "props": {"outputDataKeys": "mxpi_modelinfer0"},
                                "next": "appsink0"},
        "appsink0": {"factory": "appsink",
                     "props": {"blocksize": "4096000"}},
    }
}


def parse_e2e_output(mist_pkg, img_w, img_h, model_w=MODEL_W, model_h=MODEL_H):
    """Parse YOLO26 e2e output (300,6) float32: cx,cy,w,h,score,class -> pixel coords.

    Letterbox geometry is available from the visionInfo preprocess record
    (keepAspectRatioScaling + paste offsets), fallback to manual computation.
    """
    import numpy as np
    results = []
    tensor_pkg_list = MxpiDataType_pb2.MxpiTensorPackageList()
    # mistPackageVec[0] -> MxpiTensorPackageList serialized
    tensor_pkg_list.ParseFromString(mist_pkg.mistPackageVec[0].memSlice)
    scale = min(model_w / img_w, model_h / img_h)
    pad_x = (model_w - img_w * scale) / 2.0
    pad_y = (model_h - img_h * scale) / 2.0
    for tensor_pkg in tensor_pkg_list.tensorPackageVec:
        for tensor in tensor_pkg.tensorVec:
            shape = list(tensor.tensorShape)  # e.g. [300, 6]
            data = tensor.dataStr  # bytes on host
            if not data:
                continue
            n_elem = 1
            for d in shape:
                n_elem *= d
            arr = np.frombuffer(data, dtype=np.float32, count=n_elem).reshape(shape)
            arr6 = arr.reshape(-1, 6)
            for row in arr6:
                cx, cy, w, h, score, cls = row.tolist()
                if score < 0.01:
                    continue
                x0 = (cx - w / 2 - pad_x) / scale
                y0 = (cy - h / 2 - pad_y) / scale
                x1 = (cx + w / 2 - pad_x) / scale
                y1 = (cy + w / 2 - pad_y) / scale
                results.append((COCO[int(cls)] if 0 <= int(cls) < len(COCO) else str(int(cls)),
                                round(score, 3),
                                (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))))
    return results




def parse_tensor_bytes(raw, img_w, img_h):
    """Parse dataserialize JSON output {"MxpiTensorPackage":[{"tensorVec":[...]}]}."""
    import base64
    import json as _json
    import numpy as np
    doc = _json.loads(raw)
    scale = min(MODEL_W / img_w, MODEL_H / img_h)
    pad_x = (MODEL_W - img_w * scale) / 2.0
    pad_y = (MODEL_H - img_h * scale) / 2.0
    results = []
    for tensor_pkg in doc.get("MxpiTensorPackage", []):
        for tensor in tensor_pkg.get("tensorVec", []):
            shape = tensor.get("tensorShape", [])
            if not shape:
                continue
            data_str = tensor.get("dataStr", "")
            if data_str:
                try:
                    data = base64.b64decode(data_str)
                except Exception:
                    data = data_str.encode("latin-1")
            else:
                # JSON can't carry binary: read from host pointer via ctypes
                ptr = int(tensor.get("tensorDataPtr", 0))
                size = int(tensor.get("tensorDataSize", 0))
                if not ptr or not size:
                    continue
                import ctypes
                buf = (ctypes.c_ubyte * size).from_address(ptr)
                data = bytes(buf)
            n_elem = 1
            for d in shape:
                n_elem *= d
            if len(data) < n_elem * 4:
                continue
            arr = np.frombuffer(data, dtype=np.float32, count=n_elem).reshape(shape)
            arr6 = arr.reshape(-1, 6)
            for row in arr6:
                cx, cy, w, h, score, cls = row.tolist()
                if score < 0.01:
                    continue
                x0 = (cx - w / 2 - pad_x) / scale
                y0 = (cy - h / 2 - pad_y) / scale
                x1 = (cx + w / 2 - pad_x) / scale
                y1 = (cy + w / 2 - pad_y) / scale
                results.append((COCO[int(cls)] if 0 <= int(cls) < len(COCO) else str(int(cls)),
                                round(score, 3),
                                (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))))
    return results


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "bus_det_310p.jpg")

    stream_mgr = StreamManagerApi()
    ret = stream_mgr.InitManager()
    assert ret == 0, "InitManager failed"

    pipeline_file = os.path.join(BASE, "mindx_pipeline.json")
    with open(pipeline_file, "w") as f:
        json.dump(PIPELINE_JSON, f, indent=2)
    ret = stream_mgr.CreateMultipleStreamsFromFile("/data2/gxw/yolo26-310p-eval/mindx_pipeline.json".encode("utf-8"))
    if ret != 0:
        print(stream_mgr.GetLastErrorMsg())
        raise RuntimeError(f"CreateMultipleStreamsFromFile failed: {ret}")

    stream_name = b"imagedecoder"  # SWIG binding: all std::string args need bytes
    data_input = MxDataInput()
    with open(img_path, "rb") as f:
        data_input.data = f.read()

    from PIL import Image
    img_w, img_h = Image.open(img_path).size

    in_plugin_id = 0
    for run in range(10):
        t0 = time.time()
        # official-sample form: SendDataWithUniqueId(streamName, inPluginId, dataInput)
        # returns uniqueId directly (negative = failure)
        unique_id = stream_mgr.SendDataWithUniqueId(stream_name, in_plugin_id, data_input)
        if unique_id < 0:
            print(stream_mgr.GetLastErrorMsg())
            raise RuntimeError(f"SendDataWithUniqueId failed: {unique_id}")
        out = stream_mgr.GetResultWithUniqueId(stream_name, unique_id, 10000)
        dt = time.time() - t0
        if out.errorCode != 0:
            print(f"[err] errorCode={out.errorCode}: {out.data}")
            raise RuntimeError("infer error")
        # out.data is serialized MxpiTensorPackageList (from appsink + dataserialize)
        raw = out.data
        if run == 0:
            doc = json.loads(raw)
            for tp in doc.get("MxpiTensorPackage", []):
                for t in tp.get("tensorVec", []):
                    ds = t.get("dataStr", "")
                    print(f"[debug] tensor keys={sorted(t.keys())}")
                    print(f"[debug] shape={t.get('tensorShape')}, dataType={t.get('tensorDataType')}, "
                          f"memType={t.get('memType')}, deviceId={t.get('deviceId')}, "
                          f"ptr={t.get('tensorDataPtr')}, size={t.get('tensorDataSize')}, dataStr len={len(ds)}")
        results = parse_tensor_bytes(raw, img_w, img_h)
        print(f"--- run {run} ({dt*1000:.1f} ms) ---")
        for cls, score, box in results[:10]:
            print(f"  {cls:<15} {score:<6} {box}")
        print(f"  total {len(results)} detections")

    stream_mgr.DestroyAllStreams()
    print("PASS")


if __name__ == "__main__":
    main()
