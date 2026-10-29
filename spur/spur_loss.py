"""
spur / spur_loss.py — SPUR loss on a YOLO11n-det head.

Reproduces the original SPUR three-stage schedule (hyper-parameters match the
original `config.py`):
  warmup  : plain average (cls + reg)
  control : per-scale-group minimax (in-group softmax adversarial weighting)
            + control term
  full    : minimax (proxy / gradient-payload decoupled) + Kendall
            uncertainty + control term

The original anchor-based center matching is replaced by YOLO11n's
TaskAlignedAssigner; the original conf/noobj term is replaced by BCE on all
anchors.
"""

import torch
import torch.nn as nn

from spur.yolocore.model import dist2bbox, make_anchors                    # noqa: E402
from spur.yolocore.loss_utils import (                                     # noqa: E402
    TaskAlignedAssigner, bbox_iou, bbox2dist)

# ---- SPUR hyper-parameters (original config.py / yolo_training_full.py) ----
K_GROUPS = 3
ETA = 3.0
LAMBDA_CONTROL = 0.005
GAMA_CONTROL = 1.0
WARMUP_EPOCHS = 20
CONTROL_EPOCHS = 80
SIGMA_REG = 0.1
NEG_WEIGHT = 1.0


def pad_gts(targets, B, device):
    """targets [N,6] -> gt_labels [B,M,1], gt_bboxes [B,M,4], mask [B,M,1]."""
    gl, gb, M = [], [], 0
    for b in range(B):
        m = targets[targets[:, 0] == b]
        M = max(M, len(m))
        gl.append(m[:, 1])
        gb.append(m[:, 2:])
    if M == 0:
        return (torch.zeros((B, 0, 1), device=device),
                torch.zeros((B, 0, 4), device=device),
                torch.zeros((B, 0, 1), device=device, dtype=torch.bool))
    gt_labels = torch.zeros((B, M, 1), device=device)
    gt_bboxes = torch.zeros((B, M, 4), device=device)
    mask = torch.zeros((B, M, 1), device=device, dtype=torch.bool)
    for b in range(B):
        n = len(gl[b])
        if n:
            gt_labels[b, :n, 0] = gl[b]
            gt_bboxes[b, :n] = gb[b]
            mask[b, :n, 0] = True
    return gt_labels, gt_bboxes, mask


def minimax(payload, proxy, norm_area, n_pos, n_groups, eta):
    """Per-scale-group adversarial weighting: in-group softmax(proxy) -> weight,
    weighted payload, sum across groups / n_pos."""
    thr = torch.linspace(0, 1.0, n_groups + 1, device=payload.device)
    total = payload.new_tensor(0.0)
    for k in range(n_groups):
        lo, hi = thr[k], thr[k + 1]
        if k == n_groups - 1:
            m = (norm_area >= lo) & (norm_area <= hi)
        else:
            m = (norm_area >= lo) & (norm_area < hi)
        if int(m.sum()) == 0:
            continue
        p = torch.softmax(proxy[m] / eta, dim=0).detach()
        total = total + (p * payload[m]).sum()
    return total / max(n_pos, 1)


class SPURLoss(nn.Module):
    def __init__(self, nc=1, strides=(8.0, 16.0, 32.0)):
        super().__init__()
        self.nc = nc
        self.strides = strides
        self.assigner = TaskAlignedAssigner(10, nc, 0.5, 6.0, strides)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.register_buffer('proj', torch.arange(16, dtype=torch.float),
                             persistent=False)
        self.epoch = 0
        self.stage = 'warmup'

    def set_epoch(self, model, criterion, epoch):
        """engine epoch_callback entry."""
        self.epoch = epoch
        if epoch < WARMUP_EPOCHS:
            self.stage = 'warmup'
        elif epoch < CONTROL_EPOCHS:
            self.stage = 'control'
        else:
            self.stage = 'full'
        print(f'[SPUR] epoch {epoch} stage={self.stage}')

    def forward(self, preds, targets):
        device = preds[0].device
        B = preds[0].shape[0]
        preds = [p.float() for p in preds]
        off = 4 * 16
        nc = self.nc

        box_l, cls_l, sig_l = [], [], []
        for p in preds:
            box_l.append(p[:, :off].reshape(B, off, -1))
            cls_l.append(p[:, off:off + nc].reshape(B, nc, -1))
            sig_l.append(p[:, off + nc:off + nc + 1].reshape(B, 1, -1))
        pred_dist = torch.cat(box_l, -1).permute(0, 2, 1)     # [B,A,64]
        pred_scores = torch.cat(cls_l, -1).permute(0, 2, 1)   # [B,A,nc]
        pred_sigma = torch.cat(sig_l, -1).permute(0, 2, 1)    # [B,A,1]
        A = pred_dist.shape[1]

        strides_t = torch.tensor(self.strides, device=device)
        anchors, stride_col = make_anchors(preds, strides_t, 0.5)
        stride = stride_col  # [A,1]

        pd = pred_dist.view(B, A, 4, 16).softmax(-1) @ self.proj.to(device)
        pred_bboxes_grid = dist2bbox(pd, anchors[None], xywh=False)

        gt_labels, gt_bboxes, mask_gt = pad_gts(targets, B, device)
        _, tbb, tscores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes_grid.detach() * stride).type(gt_bboxes.dtype),
            anchors * stride, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(float(tscores.sum()), 1.0)

        # Background suppression (replaces original conf/noobj; BCE on all anchors)
        neg = (self.bce(pred_scores, tscores).sum() / target_scores_sum) * NEG_WEIGHT

        n_pos = int(fg_mask.sum())
        stage_loss = preds[0].new_tensor(0.0)
        loc_log = preds[0].new_tensor(0.0)
        if n_pos:
            s_map = stride.view(-1)[None].expand(B, A)
            t_grid = tbb[fg_mask] / s_map[fg_mask].unsqueeze(-1)
            p_grid = pred_bboxes_grid[fg_mask]

            cls_inst = self.bce(pred_scores[fg_mask], tscores[fg_mask]).mean(-1)
            ciou = bbox_iou(p_grid, t_grid)
            reg_inst = (1.0 - ciou)

            # control: 1/sqrt(area_grid) * center squared error
            tw = t_grid[:, 2] - t_grid[:, 0]
            th = t_grid[:, 3] - t_grid[:, 1]
            area_grid = (tw * th).clamp(min=0)
            wgt = 1.0 / torch.sqrt(area_grid + 1e-6)
            t_cx = (t_grid[:, 0] + t_grid[:, 2]) / 2
            t_cy = (t_grid[:, 1] + t_grid[:, 3]) / 2
            p_cx = (p_grid[:, 0] + p_grid[:, 2]) / 2
            p_cy = (p_grid[:, 1] + p_grid[:, 3]) / 2
            control_inst = wgt * ((p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2)

            # Normalised area (letterbox input side, inferred from P3 / stride0)
            S = preds[0].shape[-1] * self.strides[0]
            tbb_p = tbb[fg_mask]
            nw = (tbb_p[:, 2] - tbb_p[:, 0]) / S
            nh = (tbb_p[:, 3] - tbb_p[:, 1]) / S
            norm_area = (nw * nh).clamp(0, 1)

            sigma = pred_sigma[fg_mask][:, 0].clamp(-7.0, 2.0)

            if self.stage == 'warmup':
                stage_loss = cls_inst.mean() + reg_inst.mean()
                loc_log = reg_inst.mean()
            elif self.stage == 'control':
                task = cls_inst + reg_inst + LAMBDA_CONTROL * control_inst
                stage_loss = minimax(task, task, norm_area, n_pos,
                                     K_GROUPS, ETA)
                loc_log = reg_inst.mean() + (LAMBDA_CONTROL * control_inst).mean()
            else:
                unc = (0.5 * torch.exp(-sigma) * reg_inst
                       + 0.5 * sigma + SIGMA_REG * sigma ** 2)
                proxy = cls_inst + reg_inst + LAMBDA_CONTROL * control_inst
                payload = (cls_inst + GAMA_CONTROL * unc
                           + LAMBDA_CONTROL * control_inst)
                stage_loss = minimax(payload, proxy, norm_area, n_pos,
                                     K_GROUPS, ETA)
                loc_log = (GAMA_CONTROL * unc).mean() + \
                          (LAMBDA_CONTROL * control_inst).mean()

        loss = stage_loss + neg
        comps = torch.stack([stage_loss.detach(), loc_log.detach(),
                             neg.detach()])
        return loss * B, comps
