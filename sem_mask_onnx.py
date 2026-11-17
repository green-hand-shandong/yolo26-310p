#!/usr/bin/env python3
# P0-1 步骤A: KITTI 0033 语义掩码生成 (yolo26n-seg ONNX @ onnxruntime CPU)
# 输出: /out/sem_masks/%010d.npz  {masks: [N,H,W] uint8 bool, cls: [N]}
# 后处理: letterbox 640 → 检测头 [1,300,38] + proto [1,32,160,160] → proto 组装 → 上采样
import os, time
import numpy as np
import cv2
import onnxruntime as ort

KITTI = os.environ.get('KITTI_SEQ', '/data0/kitti/2011_09_30/2011_09_30_drive_0033_sync')
OUT = os.environ.get('OUT', '/out/sem_masks')
ONNX = os.environ.get('ONNX', '/out/yolo26n-seg.onnx')
IMG_DIR = os.environ.get('IMG_DIR', 'image_00')
os.makedirs(OUT, exist_ok=True)

sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]
IN_H, IN_W = 640, 640
# COCO 动态类: person0 bicycle1 car2 motorcycle3 bus5 train6 truck7
DYNAMIC = {0, 1, 2, 3, 5, 6, 7}
CONF_TH = 0.25

def letterbox(img, new=640):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), 114, np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top+nh, left:left+nw] = rs
    return canvas, r, left, top

imgs = sorted(os.listdir(f'{KITTI}/{IMG_DIR}/data'))
t0 = time.time(); n = 0
for fn in imgs:
    fid = int(fn.split('.')[0])
    outp = f'{OUT}/{fn.replace(".png", ".npz")}'
    if os.path.exists(outp):
        continue
    img = cv2.imread(f'{KITTI}/{IMG_DIR}/data/{fn}')
    lb, r, dx, dy = letterbox(img)
    x = lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    outs = sess.run(out_names, {in_name: x})
    det, proto = outs[0], outs[1]           # det [1,300,38], proto [1,32,160,160]
    det = det[0]
    # 解析: [cx,cy,w,h, scores..., mask_coeffs(32)] — 找 38 列的构成: 4 box + 80 cls 或 4+1obj+80?
    # yolo26-seg导出: 4 + num_cls(80) + 32 = 116? 38 = 4 + 2? 先自适应: mask_coeffs 取最后32列
    conf = det[:, 4]
    cls_id = det[:, 5].astype(int)
    coef = det[:, 6:]
    keep = (conf > CONF_TH) & np.isin(cls_id, list(DYNAMIC))
    masks_out = []
    clss_out = []
    if keep.any():
        ph, pw = proto.shape[2], proto.shape[3]
        P = proto[0].reshape(proto.shape[1], -1)  # [32, 160*160]
        for i in np.where(keep)[0]:
            m = (coef[i] @ P).reshape(ph, pw)
            m = 1 / (1 + np.exp(-m))
            # proto(160x160) → 640x640(letterbox 空间) → 裁内容区 → 原图尺寸
            m640 = cv2.resize(m, (640, 640))
            ih, iw = img.shape[:2]
            nh, nw = int(round(ih * r)), int(round(iw * r))
            y0, x0 = (640 - nh) // 2, (640 - nw) // 2
            m_crop = m640[y0:y0+nh, x0:x0+nw]
            m_full = cv2.resize(m_crop, (iw, ih))
            masks_out.append((m_full > 0.5).astype(np.uint8))
            clss_out.append(cls_id[i])
    if masks_out:
        np.savez_compressed(outp, masks=np.stack(masks_out), cls=np.array(clss_out, np.int64))
    else:
        np.savez_compressed(outp, masks=np.zeros((0, img.shape[0], img.shape[1]), np.uint8), cls=np.zeros(0, np.int64))
    n += 1
    if n % 100 == 0:
        el = time.time() - t0
        print(f'{n}/{len(imgs)} {el/n*1000:.0f} ms/frame', flush=True)
print(f'DONE {n} in {time.time()-t0:.0f}s', flush=True)
