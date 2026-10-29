"""Standard YOLO-format dataset (images/<split> + labels/<split>).

Expected layout (matches ultralytics YOLOv5/v8/v11 convention)::

    <data_root>/
        images/train/xxx.jpg
        images/val/yyy.jpg
        labels/train/xxx.txt
        labels/val/yyy.txt

Each label file is one line per object: ``cls cx cy w h`` (normalized 0..1).

Output of ``__getitem__`` matches the SPUR loss expectations:
    image  : float32 tensor [3, S, S] in [0,1], letterboxed to S×S
    targets: [N, 5] = (cls, x1, y1, x2, y2) in letterbox pixel coords
"""

import os
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PAD_VALUE = 114
IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')


# ============================================================
# Image / label scanning
# ============================================================
def img_list(split_dir):
    out = []
    for ext in IMG_EXTS:
        out += glob.glob(os.path.join(split_dir, ext))
        out += glob.glob(os.path.join(split_dir, '**', ext), recursive=True)
    return sorted(set(out))


def img_to_label_path(img_path, images_root, labels_root):
    """images_root/.../xxx.jpg -> labels_root/.../xxx.txt (same relative path)."""
    rel = os.path.relpath(img_path, images_root)
    rel_noext = os.path.splitext(rel)[0] + '.txt'
    return os.path.join(labels_root, rel_noext)


def load_label(txt_path):
    """-> [N,5] = cls, cx, cy, w, h normalized; empty -> (0,5)."""
    if not os.path.isfile(txt_path):
        return np.zeros((0, 5), np.float32)
    rows = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 5:
                continue
            rows.append([float(x) for x in parts[:5]])
    if not rows:
        return np.zeros((0, 5), np.float32)
    return np.asarray(rows, np.float32)


# ============================================================
# Letterbox + augmentation
# ============================================================
def letterbox(img, new_size, boxes=None):
    """Equal-ratio resize + pad to new_size×new_size.

    boxes: [N,5] (cls,cx,cy,w,h) normalized, or None.
    Returns img, boxes([M,5] cls,x1,y1,x2,y2 pixel), ratio, (dw,dh).
    """
    h0, w0 = img.shape[:2]
    ratio = min(new_size / w0, new_size / h0)
    nw, nh = int(round(w0 * ratio)), int(round(h0 * ratio))
    dw = (new_size - nw) / 2
    dh = (new_size - nh) / 2

    if (nw, nh) != (w0, h0):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))

    if boxes is None or len(boxes) == 0:
        return img, np.zeros((0, 5), np.float32), ratio, (dw, dh)

    cls = boxes[:, 0]
    cx, cy, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    cx = cx * w0 * ratio + dw
    cy = cy * h0 * ratio + dh
    w = w * w0 * ratio
    h = h * h0 * ratio
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    out = np.stack([cls, x1, y1, x2, y2], 1)
    out[:, 1:] = np.clip(out[:, 1:], 0, new_size - 1e-3)
    return img, out, ratio, (dw, dh)


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    np.copyto(img, cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))


# ============================================================
# Dataset
# ============================================================
class YOLODataset(Dataset):
    """Reads images/<split> + labels/<split> in standard YOLO format."""

    def __init__(self, data_root, split='train', img_size=640, augment=False):
        self.data_root = data_root
        self.images_root = os.path.join(data_root, 'images', split)
        self.labels_root = os.path.join(data_root, 'labels', split)
        self.img_size = img_size
        self.augment = augment
        self.files = img_list(self.images_root)
        if not self.files:
            raise FileNotFoundError(
                f'No images under {self.images_root}. '
                f'Expected layout: <data_root>/images/{split}/*.jpg')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ip = self.files[idx]
        img = cv2.imread(ip)
        if img is None:
            # graceful skip: return a gray placeholder
            img = np.full((self.img_size, self.img_size, 3), PAD_VALUE, np.uint8)
            boxes = np.zeros((0, 5), np.float32)
        else:
            txt = img_to_label_path(ip, self.images_root, self.labels_root)
            boxes = load_label(txt)
            img, boxes, _, _ = letterbox(img, self.img_size, boxes)

        if self.augment:
            if np.random.random() < 0.5:
                img = img[:, ::-1, :]
                if len(boxes):
                    x1 = boxes[:, 1].copy()
                    x2 = boxes[:, 3].copy()
                    boxes[:, 1] = self.img_size - x2
                    boxes[:, 3] = self.img_size - x1
            augment_hsv(img)

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR -> RGB
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        if len(boxes):
            targets = torch.from_numpy(boxes).float()
            keep = (targets[:, 3] > targets[:, 1] + 1e-3) & \
                   (targets[:, 4] > targets[:, 2] + 1e-3)
            targets = targets[keep]
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)
        return img_t, targets


def collate_images(batch):
    imgs, ts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    out = []
    for b, t in enumerate(ts):
        if len(t):
            bi = torch.full((len(t), 1), b, dtype=torch.float32)
            out.append(torch.cat([bi, t], 1))   # b,cls,x1,y1,x2,y2
    targets = torch.cat(out, 0) if out else torch.zeros((0, 6), dtype=torch.float32)
    return imgs, targets


def create_dataloader(data_root, split='train', img_size=640, batch_size=8,
                      augment=False, shuffle=None, workers=2):
    ds = YOLODataset(data_root, split, img_size, augment)
    if shuffle is None:
        shuffle = augment
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=workers, collate_fn=collate_images,
                        drop_last=augment, pin_memory=False)
    return loader
