"""YOLO11n training-side utilities.

Includes:
- `DetectRaw` — training head that returns raw per-scale feature maps (no decode).
- `bias_init`, `adapt_bn` — weight init that matches `ultralytics` defaults.
- `bbox_iou` — CIoU on xyxy boxes.
- `TaskAlignedAssigner` — the official anchor-free target assigner.
- `bbox2dist`, `dfl_loss` — distribution focal loss helpers.
- `V8DetectionLoss` — the standard yolo11n loss (not used by SPUR itself, but
  handy for an ablated / plain-yolo11n baseline).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Detect, dist2bbox, make_anchors

REG_MAX = 16
STRIDES = (8.0, 16.0, 32.0)
BOX_GAIN = 7.5
CLS_GAIN = 0.5
DFL_GAIN = 1.5
EPS = 1e-9


# ============================================================
# Training head
# ============================================================
class DetectRaw(Detect):
    """forward -> list[Tensor] each [B, 4*reg_max + nc, H, W] (no decode)."""

    def forward(self, x):
        return [torch.cat([self.cv2[i](x[i]), self.cv3[i](x[i])], dim=1)
                for i in range(self.nl)]


def bias_init(head):
    """Match ultralytics `Detect.bias_init`: box-last-layer bias=2; cls bias = prior."""
    for i, (a, b) in enumerate(zip(head.cv2, head.cv3)):
        a[-1].bias.data[:] = 2.0
        b[-1].bias.data[: head.nc] = math.log(
            5 / head.nc / (640 / head.stride[i]) ** 2)


def adapt_bn(model, eps=1e-3, momentum=0.03):
    """Match ultralytics default BN eps=1e-3, momentum=0.03."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = eps
            m.momentum = momentum
    return model


# ============================================================
# CIoU
# ============================================================
def bbox_iou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    c_x = (b1_x2 + b1_x1) / 2 - (b2_x2 + b2_x1) / 2
    c_y = (b1_y2 + b1_y1) / 2 - (b2_y2 + b2_y1) / 2
    d2 = c_x ** 2 + c_y ** 2
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return (iou - d2 / c2 - alpha * v).squeeze(-1)


# ============================================================
# Task-Aligned Assigner
# ============================================================
class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0,
                 strides=(8, 16, 32)):
        super().__init__()
        self.topk = topk
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.strides = list(strides)
        self.stride_val = self.strides[1] if len(self.strides) > 1 else self.strides[0]

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        B, A, nc = pd_scores.shape
        M = gt_bboxes.shape[1]
        device = pd_scores.device
        if M == 0:
            return (torch.zeros((B, A), device=device, dtype=torch.long),
                    torch.zeros((B, A, 4), device=device),
                    torch.zeros((B, A, nc), device=device),
                    torch.zeros((B, A), device=device, dtype=torch.bool),
                    torch.zeros((B, A), device=device, dtype=torch.long))

        mask_in_gts = self._candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self._box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self._topk(align_metric,
                               mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        target_gt_idx, fg_mask, mask_pos = self._highest_overlaps(
            mask_pos, overlaps, M)
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask, B, M)

        align_metric = align_metric * mask_pos
        pos_align = align_metric.amax(dim=-1, keepdim=True)
        pos_ov = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm = (align_metric * pos_ov / (pos_align + EPS)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _candidates_in_gts(self, centers, gt_bboxes, mask_gt):
        xywh = self._xyxy2xywh(gt_bboxes)
        small = xywh[..., 2:] < self.strides[0]
        fill = torch.tensor(self.stride_val, dtype=xywh.dtype, device=xywh.device)
        xywh[..., 2:] = torch.where((small * mask_gt).bool(), fill, xywh[..., 2:])
        bboxes = self._xywh2xyxy(xywh)
        B, M, _ = bboxes.shape
        A = centers.shape[0]
        lt, rb = bboxes.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat([centers[None] - lt, rb - centers[None]], 2).view(B, M, A, -1)
        return deltas.amin(3) > EPS

    def _box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        B, A, nc = pd_scores.shape
        M = gt_bboxes.shape[1]
        device = pd_scores.device
        mg = mask_gt.bool()
        overlaps = torch.zeros((B, M, A), device=device)
        scores = torch.zeros((B, M, A), device=device)
        ind0 = torch.arange(B, device=device).view(-1, 1).expand(-1, M)
        ind1 = gt_labels.squeeze(-1).long()
        cls_per_anchor = pd_scores.permute(0, 2, 1)[ind0, ind1]
        scores[mg] = cls_per_anchor[mg]
        pd_e = pd_bboxes.unsqueeze(1).expand(-1, M, -1, -1)[mg]
        gt_e = gt_bboxes.unsqueeze(2).expand(-1, -1, A, -1)[mg]
        overlaps[mg] = bbox_iou(gt_e, pd_e).clamp(0)
        align = scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align, overlaps

    def _topk(self, metrics, topk_mask):
        topk_metrics, topk_idx = metrics.topk(self.topk, dim=-1)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > EPS).expand_as(topk_idx)
        topk_idx.masked_fill_(~topk_mask, 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(topk_idx[..., :1], dtype=torch.int8)
        for k in range(self.topk):
            count.scatter_add_(-1, topk_idx[..., k:k + 1], ones)
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)

    def _highest_overlaps(self, mask_pos, overlaps, M):
        fg = mask_pos.sum(-2)
        if fg.max() > 1:
            multi = (fg.unsqueeze(1) > 1).expand(-1, M, -1)
            idx = overlaps.argmax(1).unsqueeze(1)
            one = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            one.scatter_(1, idx, 1)
            mask_pos = torch.where(multi, one, mask_pos)
            fg = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg, mask_pos

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, B, M):
        batch_ind = torch.arange(B, device=gt_labels.device)[..., None]
        flat_idx = (target_gt_idx + batch_ind * M)
        target_labels = gt_labels.long().flatten()[flat_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]
        target_labels.clamp_(0)
        target_scores = torch.zeros((B, target_labels.shape[1], self.nc),
                                     dtype=torch.float, device=gt_labels.device)
        target_scores.scatter_(2, target_labels[..., None], 1)
        fg_rep = fg_mask[..., None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_rep > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    @staticmethod
    def _xyxy2xywh(b):
        x1, y1, x2, y2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)

    @staticmethod
    def _xywh2xyxy(b):
        x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)


# ============================================================
# DFL
# ============================================================
def dfl_loss(pred_dist, target):
    target = target.clamp(0, REG_MAX - 1 - 0.01)
    tl = target.long()
    tr = tl + 1
    wl = (tr.float() - target)
    wr = 1 - wl
    n, s = tl.shape
    ce = F.cross_entropy(pred_dist.view(-1, REG_MAX), tl.view(-1),
                         reduction='none').view(n, s) * wl + \
         F.cross_entropy(pred_dist.view(-1, REG_MAX), tr.view(-1),
                         reduction='none').view(n, s) * wr
    return ce.mean(-1)


def bbox2dist(anchor, bbox, reg_max=REG_MAX - 1):
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat([anchor - x1y1, x2y2 - anchor], -1).clamp(0, reg_max)


# ============================================================
# Standard V8/11 loss (reference; SPUR uses its own SPURLoss)
# ============================================================
class V8DetectionLoss(nn.Module):
    def __init__(self, nc=1, reg_max=REG_MAX, strides=STRIDES):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.assigner = TaskAlignedAssigner(10, nc, 0.5, 6.0, strides)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.register_buffer('proj', torch.arange(reg_max, dtype=torch.float),
                             persistent=False)

    def forward(self, preds, targets):
        device = preds[0].device
        B = preds[0].shape[0]
        preds = [p.float() for p in preds]

        dist_l, score_l = [], []
        for p in preds:
            dist_l.append(p[:, :4 * self.reg_max].reshape(B, 4 * self.reg_max, -1))
            score_l.append(p[:, 4 * self.reg_max:].reshape(B, self.nc, -1))
        pred_dist = torch.cat(dist_l, -1).permute(0, 2, 1)
        pred_scores = torch.cat(score_l, -1).permute(0, 2, 1)
        A = pred_dist.shape[1]

        strides_t = torch.tensor(self.strides, device=device)
        anchors, stride_col = make_anchors(preds, strides_t, 0.5)
        stride = stride_col

        pd = pred_dist.view(B, A, 4, self.reg_max).softmax(-1) @ self.proj.to(device)
        pred_bboxes_grid = dist2bbox(pd, anchors[None], xywh=False)

        M, gl, gb = 0, [], []
        for b in range(B):
            m = targets[targets[:, 0] == b]
            M = max(M, len(m))
            gl.append(m[:, 1])
            gb.append(m[:, 2:])
        if M == 0:
            gt_labels = torch.zeros((B, 0, 1), device=device)
            gt_bboxes = torch.zeros((B, 0, 4), device=device)
            mask_gt = torch.zeros((B, 0, 1), device=device, dtype=torch.bool)
        else:
            gt_labels = torch.zeros((B, M, 1), device=device)
            gt_bboxes = torch.zeros((B, M, 4), device=device)
            mask_gt = torch.zeros((B, M, 1), device=device, dtype=torch.bool)
            for b in range(B):
                n = len(gl[b])
                if n:
                    gt_labels[b, :n, 0] = gl[b]
                    gt_bboxes[b, :n] = gb[b]
                    mask_gt[b, :n, 0] = True

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes_grid.detach() * stride).type(gt_bboxes.dtype),
            anchors * stride, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(float(target_scores.sum()), 1.0)
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        loss_box = preds[0].new_tensor(0.0)
        loss_dfl = preds[0].new_tensor(0.0)
        nfg = int(fg_mask.sum())
        if nfg:
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            s_map = stride.view(-1).view(1, A).expand(B, A)
            t_grid = target_bboxes[fg_mask] / s_map[fg_mask].unsqueeze(-1)
            p_grid = pred_bboxes_grid[fg_mask]
            iou = bbox_iou(p_grid, t_grid)
            loss_box = ((1.0 - iou.unsqueeze(-1)) * weight).sum() / target_scores_sum
            target_ltrb = bbox2dist(anchors[None].expand(B, A, 2)[fg_mask],
                                    t_grid, REG_MAX - 1)
            pd_fg = pred_dist[fg_mask].view(-1, self.reg_max)
            loss_dfl = (dfl_loss(pd_fg.view(-1, 4, self.reg_max), target_ltrb)
                        .unsqueeze(-1) * weight).sum() / target_scores_sum

        comps = torch.stack([BOX_GAIN * loss_box, CLS_GAIN * loss_cls,
                             DFL_GAIN * loss_dfl]).detach()
        loss = BOX_GAIN * loss_box + CLS_GAIN * loss_cls + DFL_GAIN * loss_dfl
        return loss * B, comps
