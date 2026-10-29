"""Smoke-test: build a tiny synthetic YOLO dataset, train 2 batches, run inference."""
import os, sys, shutil
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# 1) Build a tiny synthetic YOLO dataset under <repo>/data_smoke
data = os.path.join(ROOT, 'data_smoke')
for split in ('train', 'val'):
    for sub in ('images', 'labels'):
        os.makedirs(os.path.join(data, sub, split), exist_ok=True)

def make_img(path, cls, cx, cy, w, h):
    img = np.full((480, 640, 3), 114, np.uint8)
    # draw a colored square at the box (in pixel coords) so it's not empty
    x1, y1 = int((cx - w/2) * 640), int((cy - h/2) * 480)
    x2, y2 = int((cx + w/2) * 640), int((cy + h/2) * 480)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.imwrite(path, img)

rng = np.random.default_rng(0)
for i in range(4):
    img_path = os.path.join(data, 'images', 'train', f't{i:03d}.jpg')
    lab_path = os.path.join(data, 'labels', 'train', f't{i:03d}.txt')
    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    w, h = rng.uniform(0.05, 0.15), rng.uniform(0.05, 0.15)
    make_img(img_path, 0, cx, cy, w, h)
    with open(lab_path, 'w') as f:
        f.write(f'0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n')
for i in range(2):
    img_path = os.path.join(data, 'images', 'val', f'v{i:03d}.jpg')
    lab_path = os.path.join(data, 'labels', 'val', f'v{i:03d}.txt')
    make_img(img_path, 0, 0.5, 0.5, 0.1, 0.1)
    with open(lab_path, 'w') as f:
        f.write('0 0.5000 0.5000 0.1000 0.1000\n')

# 2) Model shape check
import torch
from spur.yolocore.model import YOLO11n
from spur.spur_model import build_spur, SpurRaw
from spur.spur_loss import SPURLoss

m = YOLO11n(nc=1)
m.eval()
m.h23.stride = torch.tensor([8., 16., 32.])
with torch.no_grad():
    out = m(torch.randn(1, 3, 640, 640))
print(f'[shape] yolo11n output {tuple(out.shape)} params={sum(p.numel() for p in m.parameters()):,}')

sm = build_spur(nc=1)
print(f'[shape] spur model params={sum(p.numel() for p in sm.parameters()):,}')
with torch.no_grad():
    raw = sm(torch.randn(1, 3, 320, 320))
print(f'[shape] spur raw preds: {[tuple(p.shape) for p in raw]}')

# 3) Train 2 batches
from spur.dataset import create_dataloader
from spur.engine import train_model
train_loader = create_dataloader(data, 'train', 320, 2, augment=True, workers=0)
val_loader = create_dataloader(data, 'val', 320, 2, augment=False, shuffle=False, workers=0)
crit = SPURLoss(nc=1)
last = train_model(sm, train_loader, crit, epochs=1, device=torch.device('cpu'),
                   amp=False, weights_dir=os.path.join(ROOT, 'weights'), tag='smoke',
                   max_batches=2, val_loader=val_loader,
                   epoch_callback=crit.set_epoch)
print('[train] last ckpt =', last)

# 4) Inference
from spur.detect import load_into_standard
from spur.infer import predict_image
inf = load_into_standard(last, nc=1)
inf.eval()
img = cv2.imread(os.path.join(data, 'images', 'val', 'v000.jpg'))
det = predict_image(inf, img, 320, torch.device('cpu'), 0.01, 0.45, nc=1)
print(f'[infer] detections on v000.jpg: {len(det)}')
print(det)
print('SMOKE OK')
