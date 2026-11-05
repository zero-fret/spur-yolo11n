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

Augmentations (train split, ``augment=True``):
    * Mosaic  (4-image composite, probability ``mosaic_prob``)
    * MixUp   (two-sample alpha blend, probability ``mixup_prob``)
    * horizontal flip (0.5)
    * HSV color jitter
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


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4, rng=None):
    rng = rng or np.random
    r = rng.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    np.copyto(img, cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))


def _crop_boxes(boxes, x1, y1, size):
    """Translate boxes by (x1,y1), drop those fully outside [0,size), clamp edges."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, 1] -= x1
    b[:, 2] -= y1
    b[:, 3] -= x1
    b[:, 4] -= y1
    # keep if intersects crop
    inter = (b[:, 3] > 0) & (b[:, 1] < size) & (b[:, 4] > 0) & (b[:, 2] < size)
    b = b[inter]
    if len(b):
        b[:, 1::2] = np.clip(b[:, 1::2], 0, size - 1e-3)
        b[:, 2::2] = np.clip(b[:, 2::2], 0, size - 1e-3)
    return b


# ============================================================
# Dataset
# ============================================================
class YOLODataset(Dataset):
    """Reads images/<split> + labels/<split> in standard YOLO format."""

    def __init__(self, data_root, split='train', img_size=640, augment=False,
                 mosaic_prob=0.0, mixup_prob=0.0, seed=None):
        self.data_root = data_root
        self.images_root = os.path.join(data_root, 'images', split)
        self.labels_root = os.path.join(data_root, 'labels', split)
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = float(mosaic_prob) if augment else 0.0
        self.mixup_prob = float(mixup_prob) if augment else 0.0
        # per-dataset local generator: independent from global np.random state
        self.g = np.random.default_rng(int(seed)) if seed is not None \
            else np.random.default_rng()
        self.files = img_list(self.images_root)
        if not self.files:
            raise FileNotFoundError(
                f'No images under {self.images_root}. '
                f'Expected layout: <data_root>/images/{split}/*.jpg')

    def __len__(self):
        return len(self.files)

    # --------------------------------------------------------
    # core sample loaders
    # --------------------------------------------------------
    def _load_letterboxed(self, idx):
        """Return (bgr uint8 S×S, boxes [N,5] cls,x1,y1,x2,y2)."""
        ip = self.files[idx]
        img = cv2.imread(ip)
        if img is None:
            img = np.full((self.img_size, self.img_size, 3), PAD_VALUE, np.uint8)
            boxes = np.zeros((0, 5), np.float32)
            return img, boxes
        txt = img_to_label_path(ip, self.images_root, self.labels_root)
        boxes = load_label(txt)
        img, boxes, _, _ = letterbox(img, self.img_size, boxes)
        return img, boxes

    def _mosaic(self, idxs):
        """Standard YOLO mosaic: 4 S×S tiles on a 2S canvas, random-center crop S."""
        s = self.img_size
        canvas = np.full((2 * s, 2 * s, 3), PAD_VALUE, np.uint8)
        all_boxes = []
        # quadrant -> canvas offset
        offsets = [(0, 0), (s, 0), (0, s), (s, s)]
        for k, idx in enumerate(idxs[:4]):
            tile, boxes = self._load_letterboxed(idx)
            ox, oy = offsets[k]
            canvas[oy:oy + s, ox:ox + s] = tile
            if len(boxes):
                b = boxes.copy()
                b[:, 1] += ox
                b[:, 3] += ox
                b[:, 2] += oy
                b[:, 4] += oy
                all_boxes.append(b)

        # random mosaic center in [s/2, 3s/2]
        cx = int(self.g.uniform(s // 2, 3 * s // 2))
        cy = int(self.g.uniform(s // 2, 3 * s // 2))
        x1 = int(np.clip(cx - s // 2, 0, s))
        y1 = int(np.clip(cy - s // 2, 0, s))
        crop = canvas[y1:y1 + s, x1:x1 + s]
        boxes = np.concatenate(all_boxes, 0) if all_boxes \
            else np.zeros((0, 5), np.float32)
        boxes = _crop_boxes(boxes, x1, y1, s)
        return crop, boxes

    def _mixup(self, img_a, boxes_a, img_b, boxes_b, alpha=(32.0, 16.0)):
        lam = self.g.beta(alpha[0], alpha[1])
        img = (img_a.astype(np.float32) * lam +
               img_b.astype(np.float32) * (1.0 - lam))
        img = np.clip(img, 0, 255).astype(np.uint8)
        boxes = []
        if len(boxes_a):
            boxes.append(boxes_a)
        if len(boxes_b):
            boxes.append(boxes_b)
        boxes = np.concatenate(boxes, 0) if boxes \
            else np.zeros((0, 5), np.float32)
        return img, boxes

    # --------------------------------------------------------
    # public API
    # --------------------------------------------------------
    def _ensure_worker_seed(self):
        """Give each DataLoader worker its own independent augmentation RNG.

        Without this, every forked/spawned worker shares the same local
        generator state and produces correlated Mosaic/MixUp choices.
        """
        info = torch.utils.data.get_worker_info()
        if info is not None and not getattr(self, '_wseeded', False):
            seed = int((info.seed + info.id) % (2 ** 31))
            self.g = np.random.default_rng(seed)
            self._wseeded = True

    def __getitem__(self, idx):
        s = self.img_size
        self._ensure_worker_seed()

        if self.augment and self.g.random() < self.mosaic_prob:
            picks = [idx] + list(self.g.choice(len(self.files), size=3,
                                                replace=True))
            img, boxes = self._mosaic(picks)
            # optional second sample for mixup
            if self.g.random() < self.mixup_prob:
                picks2 = list(self.g.choice(len(self.files), size=4,
                                            replace=True))
                img2, boxes2 = self._mosaic(picks2)
                img, boxes = self._mixup(img, boxes, img2, boxes2)
        else:
            img, boxes = self._load_letterboxed(idx)
            if self.augment and self.g.random() < self.mixup_prob:
                idx2 = int(self.g.integers(0, len(self.files)))
                img2, boxes2 = self._load_letterboxed(idx2)
                img, boxes = self._mixup(img, boxes, img2, boxes2)

        if self.augment:
            if self.g.random() < 0.5:
                img = img[:, ::-1, :]
                if len(boxes):
                    x1 = boxes[:, 1].copy()
                    x2 = boxes[:, 3].copy()
                    boxes[:, 1] = s - x2
                    boxes[:, 3] = s - x1
            augment_hsv(img, rng=self.g)

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
                      augment=False, shuffle=None, workers=2,
                      mosaic_prob=0.0, mixup_prob=0.0, seed=None,
                      generator=None):
    ds = YOLODataset(data_root, split, img_size, augment,
                     mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, seed=seed)
    if shuffle is None:
        shuffle = augment
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=workers, collate_fn=collate_images,
                        drop_last=augment, pin_memory=False,
                        worker_init_fn=_worker_init, generator=generator)
    return loader


def _worker_init(worker_id):
    # tie worker RNG to base seed captured via seed.py
    from spur.seed import worker_init_fn
    worker_init_fn(worker_id)
