"""YOLO26n AMCT PTQ INT8 量化（classic graph-based API）

流程: create_quant_config -> quantize_model(插量化算子) -> 校准推理 -> save_model
产出: fake_quant ONNX(ORT精度仿真) + deploy ONNX(ATC部署) + record(量化因子)
部署: deploy ONNX -> atc(8.2.RC1, libamctacl) -> 310P3 OM
"""
import os
import sys

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
sys.path.insert(0, '/data2/gxw/amct')

import glob
import torch
import cv2
import numpy as np
import amct_pytorch as amct

WORK = '/data2/gxw/yolo26-310p-eval/amct_out'
os.makedirs(WORK, exist_ok=True)


def build_model():
    """复刻 ultralytics exporter 的导出状态: fuse + Detect.export 模式"""
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect

    m = YOLO('/data2/gxw/yolo26-310p-eval/yolo26n.pt')
    model = m.model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    # 必须在 fuse 之前: Detect.fuse() 按 end2end 选择保留 one2one(e2e) 分支
    model.end2end = True
    model = model.fuse(imgsz=640)
    for mod in model.modules():
        if isinstance(mod, Detect):
            mod.dynamic = False
            mod.export = True
            mod.format = 'onnx'
            mod.max_det = 300
    return model


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return torch.from_numpy(x.transpose(2, 0, 1)[None])


def calib_data(n=300):
    imgs = sorted(glob.glob('/data2/gxw/mot_datasets/MOT17/train/*/img1/*.jpg'))
    step = max(1, len(imgs) // n)
    return [preprocess(cv2.imread(p)) for p in imgs[::step][:n]]


def main():
    torch.manual_seed(0)
    model = build_model()
    dummy = torch.randn(1, 3, 640, 640)

    config_file = f'{WORK}/config.json'
    amct.create_quant_config(
        config_file=config_file,
        model=model,
        input_data=(dummy,),
        skip_layers=None,
        batch_num=1,
        activation_offset=True,
    )
    print('[1/4] config ->', config_file)

    record_file = f'{WORK}/record.txt'
    modified_onnx = f'{WORK}/modified_model.onnx'
    calib_model = amct.quantize_model(
        config_file,
        modified_onnx,
        record_file,
        model,
        (dummy,),
        input_names=['images'],
        output_names=['output0'],
    )
    print('[2/4] quantize_model done, calibrating...')

    data = calib_data(300)
    with torch.no_grad():
        for i, x in enumerate(data):
            calib_model(x)
            if (i + 1) % 50 == 0:
                print(f'  calib {i + 1}/{len(data)}')

    amct.save_model(
        modfied_onnx_file=modified_onnx,
        record_file=record_file,
        save_path=f'{WORK}/yolo26n_int8',
    )
    print('[3/4] save_model done')

    fake = glob.glob(f'{WORK}/*fake*onnx') + glob.glob(f'{WORK}/*fake*.onnx')
    deploy = glob.glob(f'{WORK}/*deploy*.onnx')
    print('[4/4] outputs:', fake, deploy)


if __name__ == '__main__':
    main()
