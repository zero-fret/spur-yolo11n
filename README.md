# SPUR-yolo11n

A from-scratch PyTorch re-implementation of **SPUR** (Scale-Partitioned Uncertainty Rectification for Robust UAV-on-UAV Interception, ICML 2026) on top of the
**YOLO11n-det** anchor-free backbone.

The original SPUR uses YOLOv4-tiny (CSPDarknet53-tiny, anchor-based, two
scales). This repository plugs the same SPUR training recipe onto YOLO11n
(P3/P4/P5, strides 8/16/32, ~2.6M parameters) and keeps all three stages and
their hyper-parameters identical to the original.

## What SPUR adds over a vanilla YOLO detector

1. **An extra `log_sigma` channel** per detection scale (zero-initialised 1×1
   conv) — carries the Kendall-style aleatoric uncertainty.
2. **A three-stage training schedule** (auto-switched by epoch):

   | Stage | Epochs | Loss |
   |---|---|---|
   | warmup | `< 20` | plain average `cls + reg` |
   | control | `20 .. 79` | per-scale-group minimax softmax reweighting (K=3 area buckets, η=3.0) + control term `λ·1/√area·center²` (λ=0.005) |
   | full | `>= 80` | proxy / gradient-payload decoupling; payload uses Kendall uncertainty `0.5·exp(-σ)·reg + 0.5·σ + 0.1·σ²` (σ clipped to [-7, 2]) |

3. The original anchor-based center-matching is replaced by YOLO11n's
   **TaskAlignedAssigner**; the original conf/noobj background term is replaced
   by BCE on all anchors.

At inference time the `log_sigma` head is dropped and the model runs as a
standard YOLO11n detector (no extra latency).

## Repository layout

```
spur-yolo11n/
├── spur/
│   ├── __init__.py
│   ├── dataset.py            # standard YOLO txt loader (images/+labels/)
│   ├── engine.py             # SGD/AdamW + warmup + cosine + AMP
│   ├── infer.py              # letterbox -> model -> NMS -> un-letterbox
│   ├── spur_model.py         # SpurRaw head + build_spur()
│   ├── spur_loss.py          # SPURLoss, three-stage minimax
│   ├── train.py              # training entry (--config yaml)
│   ├── detect.py             # inference entry
│   └── yolocore/
│       ├── __init__.py
│       ├── model.py          # YOLO11n backbone + neck + Detect
│       └── loss_utils.py     # DetectRaw, TaskAlignedAssigner, CIoU, DFL
├── configs/
│   └── spur_uav.yaml         # example dataset config
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Install

```bash
git clone https://github.com/zero-fret/spur-yolo11n.git
cd spur-yolo11n
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Prepare data (standard YOLO format)

```
data/
├── images/
│   ├── train/   *.jpg
│   └── val/     *.jpg
└── labels/
    ├── train/   *.txt     # cls cx cy w h (normalized 0..1)
    └── val/     *.txt
```

Point `configs/spur_uav.yaml` at it:

```yaml
data_root: data
nc: 3
names: [zigzag, straight, wave]
```

## Train

```powershell
# Full three-stage run (300 epochs, auto-switches warmup/control/full)
python -m spur.train --config configs/spur_uav.yaml --img 640 --batch 16 --epochs 300

# Smoke test (2 batches, CPU-friendly)
python -m spur.train --config configs/spur_uav.yaml --img 320 --batch 2 --epochs 1 --max-batches 2
```

Checkpoints are saved to `weights/spur_last.pt` at the end of every epoch.

## Inference

```powershell
python -m spur.detect --config configs/spur_uav.yaml `
    --source path/to/images `
    --weights weights/spur_last.pt `
    --img-size 640 --conf 0.25 --save-dir out/
```

## Evaluation

```powershell
python -m spur.eval_map --data path/to/yolo_dataset --nc 1 `
    --weights weights/spur_last.pt --img-size 640
```

This computes mAP@0.5 (VOC 11-point), mAP@[0.5:0.95] (10 IoU thresholds),
Precision / Recall at conf=0.25, and scale-wise AP@0.5 on COCO area buckets
(small < 32², medium 32²–96², large ≥ 96²).

### Reported numbers (UAV-MultiView, val split)

Trained on an RTX 4050 Laptop (6.4 GB), 120 epochs, batch=8, img=640,
Adam + cosine, AMP. Only horizontal flip and HSV augmentation are enabled
in this reproduction (Mosaic / MixUp from the original recipe are not
implemented yet).

| Metric | This repo (yolo11n, 640, 120 ep) | Original SPUR (CSPDarknet-tiny, 416, 300 ep) |
|---|---|---|
| mAP@0.5 | **90.38** | 96.10 |
| mAP@[0.5:0.95] | **54.50** | 60.49 |
| Precision @ conf=0.25 | **93.84** | — |
| Recall @ conf=0.25 | **92.33** | — |

Scale-wise AP@0.5 (this repo, COCO area buckets):

| Bucket | GT count | AP@0.5 |
|---|---|---|
| small  (<32²) | 762 | 62.99 |
| medium (32²–96²) | 910 | 42.79 |
| large  (≥96²) | 374 | 16.71 |

The ~6 pp gap to the original is explained by (a) 120 vs 300 epochs,
(b) missing Mosaic/MixUp, and (c) the backbone swap from CSPDarknet-tiny
to YOLO11n.

## Hyper-parameters (match the original SPUR `config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `K_GROUPS` | 3 | scale-area buckets in minimax |
| `ETA` | 3.0 | softmax temperature inside a bucket |
| `LAMBDA_CONTROL` | 0.005 | weight on the 1/√area center² control term |
| `GAMA_CONTROL` | 1.0 | weight on Kendall uncertainty in payload |
| `WARMUP_EPOCHS` | 20 | stage-1 length |
| `CONTROL_EPOCHS` | 80 | stage-2 boundary (>=80 -> full) |
| `SIGMA_REG` | 0.1 | L2 penalty on log_sigma |
| `NEG_WEIGHT` | 1.0 | BCE background weight |

These live at the top of `spur/spur_loss.py`.

## Citation

If you use this code, cite the original SPUR paper (ICML 2026) and, optionally,
this reproduction. See `CITATION.cff`.

## License

GPLv3. 
The original SPUR paper and its YOLOv4-tiny code are the property of
their respective authors; this repository is an independent re-implementation
and ships no original SPUR source code.
