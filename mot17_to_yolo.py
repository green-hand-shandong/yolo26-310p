"""MOT17 → YOLO 检测格式转换（小规模验证子集）

MOT17 train: 7 序列 × 3 检测器变体（DPM/FRCNN/SDP 同图不同标注），只取 SDP 变体。
行人 gt 类别 1 → YOLO 类别 0。可见率 <0.3 或遮挡 tag=2 的框跳过。
输出: /data2/gxw/datasets/mot17_yolo/{images,labels}/{train,val}
"""
import os
import glob
import cv2

SRC = '/data2/gxw/mot_datasets/MOT17/train'
DST = '/data2/gxw/datasets/mot17_yolo'
VAL_SEQS = {'MOT17-09-SDP'}  # 留一个序列做验证

def convert_seq(seq_dir, split):
    seq = os.path.basename(seq_dir)
    img_dir = os.path.join(seq_dir, 'img1')
    gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
    if not os.path.exists(gt_file):
        return 0
    # 读标注
    boxes = {}
    with open(gt_file) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 7:
                continue
            frame, _, x, y, w, h, conf, cls, vis = p[:9]
            if int(cls) != 1:  # 只保留行人
                continue
            if float(vis) < 0.3 or int(float(conf)) == 0:
                continue
            boxes.setdefault(int(frame), []).append(
                (float(x), float(y), float(w), float(h)))
    # 写图和标签
    n_img, n_box = 0, 0
    for img_path in sorted(glob.glob(os.path.join(img_dir, '*.jpg'))):
        frame = int(os.path.basename(img_path).split('.')[0])
        if frame not in boxes:
            continue
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        stem = f'{seq}_{frame:06d}'
        cv2.imwrite(f'{DST}/images/{split}/{stem}.jpg', img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(f'{DST}/labels/{split}/{stem}.txt', 'w') as f:
            for (x, y, w, h) in boxes[frame]:
                cx, cy = x + w / 2, y + h / 2
                f.write(f'0 {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}\n')
                n_box += 1
        n_img += 1
    return n_img, n_box


if __name__ == '__main__':
    for split in ('train', 'val'):
        os.makedirs(f'{DST}/images/{split}', exist_ok=True)
        os.makedirs(f'{DST}/labels/{split}', exist_ok=True)
    ti = tb = vi = vb = 0
    for seq_dir in sorted(glob.glob(f'{SRC}/*SDP')):
        seq = os.path.basename(seq_dir)
        split = 'val' if seq in VAL_SEQS else 'train'
        n_img, n_box = convert_seq(seq_dir, split)
        if split == 'train':
            ti += n_img; tb += n_box
        else:
            vi += n_img; vb += n_box
        print(f'{seq}: {n_img} imgs {n_box} boxes -> {split}')
    print(f'TOTAL train: {ti} imgs / {tb} boxes | val: {vi} imgs / {vb} boxes')

    with open(f'{DST}/mot17.yaml', 'w') as f:
        f.write(f"""path: {DST}
train: images/train
val: images/val
names:
  0: person
""")
    print('dataset yaml:', f'{DST}/mot17.yaml')
