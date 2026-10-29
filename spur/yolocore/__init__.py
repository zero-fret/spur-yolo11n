"""YOLO11n core: backbone + neck + Detect head (anchor-free).

A from-scratch PyTorch re-implementation of the YOLO11n detection network,
parametrically identical to `ultralytics` at the time of authoring. Used as the
backbone / head that SPUR wraps with its extra log_sigma channel and three-stage
loss.
"""

from .model import (
    YOLO11n,
    Detect,
    Conv,
    DWConv,
    C2f,
    C3k,
    C3k2,
    SPPF,
    C2PSA,
    DFL,
    dist2bbox,
    make_anchors,
)
