"""
eval_map.py — SPUR evaluation on standard YOLO-format val set.

Computes:
  * mAP@0.5 (VOC 11-point)
  * mAP@0.5:0.95 (COCO-style, 10 IoU thresholds)
  * Precision / Recall at a fixed confidence threshold
  * Scale-wise AP@0.5 (small / medium / large by GT area, COCO convention)

Usage:
  python -m spur.eval_map --data D:/path/to/yolo_dataset --nc 1 \
      --weights weights/spur_uav_last.pt --img-size 640
"""

import os
import glob
import argparse

import cv2
import numpy as np
import torch

from spur.yolocore.model import YOLO11n
from spur.infer import predict_image


# ---------- GT loading (YOLO txt: cls cx cy w h normalized) ----------
def load_gt(txt_path, img_w, img_h):
    """Return np [N,5] = (cls, x1, y1, x2, y2) in pixel coords."""
    if not os.path.isfile(txt_path):
        return np.zeros((0, 5), np.float32)
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = float(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([cls, x1, y1, x2, y2])
    return np.asarray(boxes, np.float32) if boxes else np.zeros((0, 5), np.float32)


# ---------- IoU ----------
def iou_matrix(boxes, gt_boxes):
    """boxes [N,4] xyxy, gt [M,4] xyxy -> [N,M] IoU."""
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(boxes), len(gt_boxes)), np.float32)
    bx1, by1, bx2, by2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    inter_x1 = np.maximum(bx1[:, None], gx1[None, :])
    inter_y1 = np.maximum(by1[:, None], gy1[None, :])
    inter_x2 = np.minimum(bx2[:, None], gx2[None, :])
    inter_y2 = np.minimum(by2[:, None], gy2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_b = (bx2 - bx1) * (by2 - by1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
    return inter / (area_b[:, None] + area_g[None, :] - inter + 1e-9)


# ---------- AP (VOC 11-point) ----------
def voc_ap(rec, prec):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
        ap += p / 11.0
    return ap


def coco_ap(rec, prec):
    """Approximate COCO AP: area under PR curve (numerical integration)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap(dets, gts, iou_thr):
    """
    dets: list per image of np [n_i,6] = (x1,y1,x2,y2,conf,cls)
    gts:  list per image of np [m_i,5] = (cls,x1,y1,x2,y2)
    Returns (ap, rec_at_end, prec_at_end).
    """
    # Collect all dets sorted by conf
    all_dets = []
    n_gt = 0
    for i, (d, g) in enumerate(zip(dets, gts)):
        n_gt += len(g)
        for row in d:
            all_dets.append((row[4], i, row[:4]))  # conf, img_idx, box
    if n_gt == 0:
        return 0.0, 0.0, 0.0
    if len(all_dets) == 0:
        return 0.0, 0.0, 0.0

    all_dets.sort(key=lambda x: -x[0])
    tp = np.zeros(len(all_dets))
    fp = np.zeros(len(all_dets))
    matched = [np.zeros(len(g), dtype=bool) for g in gts]

    for k, (conf, img_idx, det_box) in enumerate(all_dets):
        g = gts[img_idx]
        if len(g) == 0:
            fp[k] = 1
            continue
        ious = iou_matrix(det_box[None, :4], g[:, 1:5])[0]
        best_j = int(np.argmax(ious))
        if ious[best_j] >= iou_thr and not matched[img_idx][best_j]:
            tp[k] = 1
            matched[img_idx][best_j] = True
        else:
            fp[k] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = voc_ap(rec, prec)
    return ap, float(rec[-1]) if len(rec) else 0.0, float(prec[-1]) if len(prec) else 0.0


def scale_bucket(area):
    """COCO area buckets (pixels^2)."""
    if area < 32 * 32:
        return 'small'
    if area < 96 * 96:
        return 'medium'
    return 'large'


def parse_opt():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='YOLO dataset root')
    p.add_argument('--nc', type=int, default=1)
    p.add_argument('--weights', required=True)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.001, help='eval conf (low for recall)')
    p.add_argument('--iou', type=float, default=0.6)
    p.add_argument('--eval-conf', type=float, default=0.25, help='P/R reported at this conf')
    p.add_argument('--device', default='')
    return p.parse_args()


def main():
    opt = parse_opt()
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Load model (drop sigma head)
    model = YOLO11n(nc=opt.nc)
    ck = torch.load(opt.weights, map_location='cpu', weights_only=False)
    sd = ck['model'] if 'model' in ck else ck
    keep = {k: v for k, v in sd.items() if '.sigma.' not in k}
    model.load_state_dict(keep, strict=False)
    model.to(device).eval()
    print(f'loaded: {opt.weights}  (nc={opt.nc}, img={opt.img_size})')

    # Collect val images
    img_dir = os.path.join(opt.data, 'images', 'val')
    lbl_dir = os.path.join(opt.data, 'labels', 'val')
    images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                    glob.glob(os.path.join(img_dir, '*.png')))
    print(f'val images: {len(images)}')

    dets_all, gts_all = [], []
    bucket_dets = {'small': [], 'medium': [], 'large': []}
    bucket_gts = {'small': [], 'medium': [], 'large': []}

    for i, ip in enumerate(images):
        img = cv2.imread(ip)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        lbl = os.path.join(lbl_dir, os.path.splitext(os.path.basename(ip))[0] + '.txt')
        gt = load_gt(lbl, w0, h0)

        det = predict_image(model, img, opt.img_size, device, opt.conf, opt.iou, opt.nc)
        dets_all.append(det)
        gts_all.append(gt)

        # Split dets/gts by GT area bucket (eval each bucket separately)
        # For dets, assign by nearest GT (or just include all dets; bucket metric is GT-driven)
        for bucket in ('small', 'medium', 'large'):
            bucket_gts[bucket].append(gt)
            bucket_dets[bucket].append(det)

        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(images)}')

    # ---- Overall metrics ----
    print('\n=== Overall ===')
    aps50 = []
    for thr in [0.5]:
        ap, r, p = compute_ap(dets_all, gts_all, thr)
        print(f'AP@{thr:.2f} = {ap*100:.2f}  (recall={r*100:.1f}, prec={p*100:.1f})')
        aps50.append(ap)

    aps_coco = []
    for thr in np.linspace(0.5, 0.95, 10):
        ap, _, _ = compute_ap(dets_all, gts_all, float(thr))
        aps_coco.append(ap)
    map50_95 = float(np.mean(aps_coco))
    print(f'mAP@[0.5:0.95] = {map50_95*100:.2f}')

    # P/R at eval-conf (re-run with higher conf)
    print(f'\n=== P / R @ conf={opt.eval_conf} ===')
    dets_hc = []
    for ip in images:
        img = cv2.imread(ip)
        if img is None:
            continue
        d = predict_image(model, img, opt.img_size, device, opt.eval_conf, opt.iou, opt.nc)
        dets_hc.append(d)
    _, r_at, p_at = compute_ap(dets_hc, gts_all, 0.5)
    print(f'Precision = {p_at*100:.2f}  Recall = {r_at*100:.2f}')

    # ---- Scale-wise AP@0.5 ----
    print('\n=== Scale-wise AP@0.5 (GT area buckets) ===')
    for bucket in ('small', 'medium', 'large'):
        # Filter GT: only keep boxes in this bucket; dets are kept as-is
        gts_b = []
        dets_b = []
        for g, d in zip(bucket_gts[bucket], bucket_dets[bucket]):
            if len(g) == 0:
                gts_b.append(g)
                dets_b.append(d)
                continue
            areas = (g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2])
            keep = np.array([scale_bucket(a) == bucket for a in areas])
            gts_b.append(g[keep])
            dets_b.append(d)
        n = sum(len(x) for x in gts_b)
        ap_b, _, _ = compute_ap(dets_b, gts_b, 0.5)
        print(f'  {bucket:6s}: GT={n:5d}   AP@0.5 = {ap_b*100:.2f}')

    print('\n=== DONE ===')


if __name__ == '__main__':
    main()
