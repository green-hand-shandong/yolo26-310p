#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Ultralytics YOLO26 to deployment-friendly end-to-end ONNX.

Usage:
    python3 export_onnx.py yolo26n.pt          # static batch=1 (for ATC bs=1)
    python3 export_onnx.py yolo26s.pt --dynamic # dynamic batch (for ATC bs>1 / dynamic)

Notes:
    - Use nms=False (default here). The nms=True export embeds NonMaxSuppression/NonZero
      ops which FAIL to compile with atc on Ascend 310P.
    - e2e output is (batch, 300, 6): x1, y1, x2, y2, score, class_id, sorted by score desc.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", help="path to yolo26n.pt / yolo26s.pt ...")
    ap.add_argument("--dynamic", action="store_true",
                    help="export with dynamic batch/H/W (needed for ATC batch>1)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        batch=args.batch,
        dynamic=args.dynamic,
        simplify=True,
        nms=False,  # keep end-to-end TopK head; embedded NMS breaks ATC on 310P
    )
    print("exported:", path)


if __name__ == "__main__":
    main()
