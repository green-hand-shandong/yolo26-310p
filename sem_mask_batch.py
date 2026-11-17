#!/usr/bin/env python3
# P0-1 步骤A: KITTI 0033 全序列语义掩码生成 (YOLO26n-seg @ NPU)
# 输入: /data0/kitti/... (需挂载) 输出: /work/sem_masks/frame_%05d.npz
import os, sys, time
import numpy as np

KITTI = os.environ.get('KITTI_SEQ', '/data0/kitti/2011_09_30/2011_09_30_drive_0033_sync')
OUT = '/work/sem_masks'
os.makedirs(OUT, exist_ok=True)

from ultralytics import YOLO
model = YOLO('/out/yolo26n-seg.pt')

# KITTI 0033 动态物体类别: car, truck, bus, person, motorcycle, bicycle(骑手), train
DYNAMIC = {2, 7, 5, 0, 3, 1, 6}  # COCO: person0 bicycle1 car2 motorcycle3 bus5 train6 truck7

imgs = sorted(os.listdir(f'{KITTI}/image_00/data'))
t0 = time.time()
n_done = 0
for i, fn in enumerate(imgs):
    outp = f'{OUT}/{fn.replace(".png", ".npz")}'
    if os.path.exists(outp):
        continue
    r = model.predict(f'{KITTI}/image_00/data/{fn}', imgsz=640, device='cpu',
                      verbose=False, conf=0.4, classes=list(DYNAMIC))
    m = r[0]
    if m.masks is None:
        np.savez_compressed(outp, masks=np.zeros((0, 4), np.float32), boxes=np.zeros((0, 4), np.float32))
    else:
        np.savez_compressed(outp,
                            masks=m.masks.data.cpu().numpy().astype(np.uint8),
                            boxes=m.boxes.xyxy.cpu().numpy(),
                            cls=m.boxes.cls.cpu().numpy())
    n_done += 1
    if n_done % 100 == 0:
        el = time.time() - t0
        print(f'{n_done}/{len(imgs)} frames, {el/n_done*1000:.0f} ms/frame', flush=True)
print(f'DONE {n_done} frames in {time.time()-t0:.0f}s', flush=True)
