"""
spur / detect.py — SPUR (YOLO11n-det) inference entry.

The `log_sigma` head in the training checkpoint is dropped; cv2/cv3 weights are
loaded into a standard YOLO11n head and decoded with standard NMS.

Usage::
  python -m spur.detect --config configs/spur_uav.yaml \
      --source path/to/images --weights weights/spur_last.pt
"""

import os
import glob
import argparse

import cv2
import torch
import yaml

from spur.yolocore.model import YOLO11n
from spur.infer import predict_image, draw


def load_config(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_into_standard(weights, nc):
    model = YOLO11n(nc=nc)
    ck = torch.load(weights, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck
    keep = {k: v for k, v in sd.items() if '.sigma.' not in k}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    assert not missing, f'missing: {missing[:5]}'
    return model


def parse_opt():
    p = argparse.ArgumentParser(description='SPUR inference')
    p.add_argument('--config', default='', help='YAML with nc / names')
    p.add_argument('--source', required=True)
    p.add_argument('--weights', default='')
    p.add_argument('--img-size', type=int, default=416)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', default='')
    p.add_argument('--save-dir', default='')
    p.add_argument('--nc', type=int, default=0)
    return p.parse_args()


def list_images(src):
    if os.path.isdir(src):
        out = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            out += glob.glob(os.path.join(src, ext))
        return sorted(out)
    return [src]


def main():
    opt = parse_opt()
    cfg = load_config(opt.config)
    nc = opt.nc or int(cfg.get('nc', 1))
    names = cfg.get('names')

    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available()
                                         else 'cpu'))
    if opt.weights and os.path.isfile(opt.weights):
        model = load_into_standard(opt.weights, nc)
        print('loaded weights:', opt.weights)
    else:
        print('WARNING: no valid --weights, using random initialization')
        model = YOLO11n(nc=nc)
    model.to(device).eval()
    if opt.save_dir:
        os.makedirs(opt.save_dir, exist_ok=True)

    for ip in list_images(opt.source):
        img = cv2.imread(ip)
        if img is None:
            continue
        det = predict_image(model, img, opt.img_size, device,
                            opt.conf, opt.iou, nc)
        print(f'{os.path.basename(ip)}: {len(det)} detections')
        if opt.save_dir:
            vis = draw(img.copy(), det, names=names)
            cv2.imwrite(os.path.join(opt.save_dir, os.path.basename(ip)), vis)


if __name__ == '__main__':
    main()
