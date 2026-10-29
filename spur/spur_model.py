"""
spur / spur_model.py — SPUR model on a YOLO11n-det backbone.

The original SPUR uses YOLOv4-tiny (anchor-based); this version uses the
YOLO11n-det anchor-free head and appends one extra `log_sigma` channel per
scale (zero-initialised 1x1 conv) to carry Kendall-style uncertainty.
"""

import torch
import torch.nn as nn

from spur.yolocore.model import YOLO11n                              # noqa: E402
from spur.yolocore.loss_utils import DetectRaw, bias_init, adapt_bn    # noqa: E402

# YOLO11n detection-layer input channels (P3/P4/P5)
HEAD_CH = (64, 128, 256)


class SpurRaw(DetectRaw):
    """Training head: per scale output [B, 4*reg_max + nc + 1, H, W]
    (last channel = log_sigma)."""

    def __init__(self, old, head_ch=HEAD_CH):
        nn.Module.__init__(self)
        self.nc, self.nl = old.nc, old.nl
        self.reg_max, self.no = old.reg_max, old.no
        self.stride = old.stride
        self.cv2, self.cv3, self.dfl = old.cv2, old.cv3, old.dfl
        # Aleatoric-uncertainty head (1x1 conv, zero-init)
        self.sigma = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in head_ch)
        for m in self.sigma:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.dynamic = False
        self.export = False
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

    def forward(self, x):
        out = []
        for i in range(self.nl):
            out.append(torch.cat([self.cv2[i](x[i]), self.cv3[i](x[i]),
                                  self.sigma[i](x[i])], dim=1))
        return out


def build_spur(nc=1):
    """Build a YOLO11n-det SPUR model (training mode, with log_sigma head)."""
    model = YOLO11n(nc=nc)
    old = model.h23
    new = SpurRaw(old)
    model.h23 = new
    adapt_bn(model)
    if torch.all(new.stride == 0):
        new.stride = torch.tensor([8.0, 16.0, 32.0])
    bias_init(new)
    return model
