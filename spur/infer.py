"""
yolo11n / infer.py —— 推理后处理（NMS + 坐标还原）

Detect 头（eval）输出：[B, 4 + nc, A]，前 4 通道为 letterbox 图上的 xywh 像素框，
其余为各类 sigmoid 概率。
"""

import cv2
import numpy as np
import torch

from .dataset import letterbox

def xywh2xyxy(b):
    x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)

def nms(boxes_xyxy, scores, labels, iou_thres):
    """cv2 NMS；boxes_xyxy [N,4] tensor，返回保留索引。"""
    if len(boxes_xyxy) == 0:
        return []
    idx = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), 0.0, iou_thres)
    if idx is None or len(idx) == 0:
        return []
    return np.array(idx).flatten().tolist()

def postprocess(pred, conf_thres=0.25, iou_thres=0.45, nc=None):
    """pred [B,4+nc,A] -> list[np]，每个 [N,6] x1,y1,x2,y2,conf,cls。"""
    B, ch, A = pred.shape
    nc = nc or ch - 4
    results = []
    for b in range(B):
        p = pred[b].T  # [A,4+nc]
        boxes = xywh2xyxy(p[:, :4])
        scores_all = p[:, 4:]
        scores, labels = scores_all.max(1)
        keep = scores > conf_thres
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        idx = nms(boxes, scores, labels, iou_thres)
        if idx:
            out = torch.cat([boxes[idx], scores[idx, None], labels[idx, None].float()], 1)
        else:
            out = torch.zeros((0, 6))
        results.append(out.cpu().numpy())
    return results

def unletterbox(boxes, ratio, dw, dh, w0, h0):
    """letterbox 像素坐标 -> 原图像素坐标。boxes [N,4+]（前 4 列 x1y1x2y2）。"""
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - dw) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - dh) / ratio
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w0 - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h0 - 1)
    return out

def predict_image(model, bgr_img, img_size=640, device='cpu',
                  conf_thres=0.25, iou_thres=0.45, nc=None):
    """对单张 BGR 图完成完整推理，返回原图坐标的 [N,6] 检测。"""
    h0, w0 = bgr_img.shape[:2]
    lb, _, ratio, (dw, dh) = letterbox(bgr_img, img_size)
    x = lb[:, :, ::-1]  # BGR->RGB
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    x = torch.from_numpy(x).float()[None] / 255.0
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
    det = postprocess(pred, conf_thres, iou_thres, nc)[0]
    return unletterbox(det, ratio, dw, dh, w0, h0)

def draw(bgr_img, det, names=None):
    for *xyxy, conf, cls in det:
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        lab = names[int(cls)] if names else str(int(cls))
        cv2.putText(bgr_img, f'{lab} {conf:.2f}', (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return bgr_img
