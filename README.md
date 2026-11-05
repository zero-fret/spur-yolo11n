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
│   ├── eval_map.py           # mAP50 / mAP50-95 / P/R / scale-wise AP
│   ├── seed.py               # reproducible seed + DataLoader worker seed
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

### Reproducibility & data augmentation

| Flag | Default | Purpose |
|---|---|---|
| `--seed` | 42 | Seeds python, numpy, torch CPU+CUDA RNG and the DataLoader shuffle generator |
| `--deterministic` | off | Forces cuDNN deterministic kernels (slower but bit-stable) |
| `--mosaic` | 0.5 | Probability of the 4-image YOLO Mosaic composite on train samples |
| `--mixup` | 0.5 | Probability of two-sample beta MixUp on train samples |
| `--resume` | — | Path to a checkpoint to continue training from; restores model, optimizer, epoch counter and augmentation config |

```powershell
# Reproducible run with Mosaic + MixUp enabled (matches original recipe probabilities)
python -m spur.train --config configs/spur_uav.yaml --img 640 --batch 8 `
    --epochs 300 --seed 42 --mosaic 0.5 --mixup 0.5

# Resume an interrupted run from its last checkpoint
python -m spur.train --config configs/spur_uav.yaml --img 640 --batch 8 `
    --epochs 300 --seed 42 --mosaic 0.5 --mixup 0.5 `
    --resume weights/spur_last.pt
```

Train-only augmentations: Mosaic (4 tiles on a 2S canvas, random-center crop),
MixUp (Beta(32,16) image blend, union labels), horizontal flip (0.5), HSV jitter.
Validation uses letterbox only (no flip / Mosaic / MixUp). The chosen seed and
augmentation probabilities are saved into every checkpoint for traceability.

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

All runs: RTX 4050 Laptop (6.4 GB), 120 epochs, batch=8, img=640, SGD + cosine,
AMP, seed 42, full three-stage schedule (warmup <20 / control <80 / full ≥80).
The two rows differ **only** in the augmentation switch, so the comparison is
apples-to-apples: same backbone, same data, same schedule, same seed, same eval
code (`spur/eval_map.py`, conf=0.001, NMS IoU=0.6, P/R at conf=0.25).

| Metric | No-aug (`flip+HSV` only) | **Mosaic 0.5 + MixUp 0.5** | Δ | Original SPUR (CSPDarknet-tiny, 416, 300 ep) |
|---|---|---|---|---|
| mAP@0.5 | 90.38 | **90.70** | +0.32 | 96.10 |
| mAP@[0.5:0.95] | 54.50 | **56.22** | **+1.72** | 60.49 |
| Precision @ conf=0.25 | 93.84 | **96.18** | **+2.34** | — |
| Recall @ conf=0.25 | 92.33 | **93.55** | **+1.22** | — |

Scale-wise AP@0.5 (COCO area buckets, identical GT counts in both runs):

| Bucket | GT count | No-aug | **Mosaic + MixUp** | Δ |
|---|---|---|---|---|
| small  (<32²) | 762 | 62.99 | **58.90** | −4.09 |
| medium (32²–96²) | 910 | 42.79 | **47.96** | **+5.17** |
| large  (≥96²) | 374 | 16.71 | **16.95** | +0.24 |

Reading of the results:

- Mosaic + MixUp buys **+1.72 mAP@[0.5:0.95]**, +2.34 P and +1.22 R at the same
  epoch budget — the gain shows up in localization quality (the stricter IoU
  metric) more than in the loose mAP@0.5.
- The improvement is **concentrated on medium objects (+5.17 AP)**, which is the
  expected effect of Mosaic: composing four images onto a 2S canvas and cropping
  back to S rescales each source object by ~0.5–1.0×, producing far more
  medium-scale instances per batch.
- **Small-object AP drops 4.09** as a direct consequence of the same mechanism —
  Mosaic shrinks small objects further, and the minimax re-weighting in the
  control stage then shifts capacity toward the medium bucket. If small-object
  accuracy is the priority for your downstream task, lower `--mosaic` (e.g. 0.3)
  or disable Mosaic for the last ~10% of epochs (the usual YOLO "close_mosaic"
  trick, not yet implemented here).
- Remaining gap to the original is explained by (a) 120 vs 300 epochs,
  (b) 640 vs 416 input, and (c) the backbone swap from CSPDarknet-tiny to YOLO11n.

### Intermediate checkpoints (ablation note)

For reference, the augmented run at epoch 75 (i.e. before reaching the `full`
stage, where the Kendall-uncertainty term activates) scored mAP@0.5 = 90.32,
mAP@[0.5:0.95] = 51.35, P = 93.83, R = 91.35, small = 37.06, medium = 51.83,
large = 17.14. It is **not** comparable to the table above — the run was only
62.5% complete and the third training stage had not started. The `full` stage
recovered most of the small-object drop (37.06 → 58.90) while keeping the
medium-object gain.

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
