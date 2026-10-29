"""
spur / train.py — SPUR (YOLO11n-det) training entry.

Usage::
  python -m spur.train --config configs/spur_uav.yaml \
      --img 640 --batch 16 --epochs 300          # full three-stage run
  python -m spur.train --config configs/spur_uav.yaml \
      --img 320 --batch 2 --max-batches 2         # smoke (2 batches)

Stage is auto-switched by epoch: <20 warmup, <80 control, >=80 full.
"""

import os
import argparse

import yaml

from spur.dataset import create_dataloader
from spur.engine import train_model
from spur.spur_model import build_spur
from spur.spur_loss import SPURLoss


def load_config(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_opt():
    p = argparse.ArgumentParser(description='SPUR (YOLO11n-det) training')
    p.add_argument('--config', default='configs/spur_uav.yaml',
                   help='YAML with data_root / nc / names')
    p.add_argument('--data', default='', help='override data_root from config')
    p.add_argument('--nc', type=int, default=0, help='override nc from config')
    p.add_argument('--img-size', type=int, default=416)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr0', type=float, default=0.01)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--tag', default='spur')
    p.add_argument('--weights-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'weights'))
    return p.parse_args()


def main():
    opt = parse_opt()
    cfg = load_config(opt.config)
    data_root = opt.data or cfg.get('data_root', 'data')
    nc = opt.nc or int(cfg.get('nc', 1))

    import torch
    device = torch.device(opt.device or ('cuda' if torch.cuda.is_available()
                                         else 'cpu'))
    print(f'=== SPUR train | device={device} | img={opt.img_size} '
          f'batch={opt.batch} | data={data_root} nc={nc} ===')

    model = build_spur(nc=nc)
    print(f'parameters: {sum(p.numel() for p in model.parameters()):,}')

    train_loader = create_dataloader(data_root, 'train', opt.img_size,
                                     opt.batch, augment=True,
                                     workers=opt.workers)
    val_loader = create_dataloader(data_root, 'val', opt.img_size,
                                   opt.batch, augment=False, shuffle=False,
                                   workers=opt.workers)
    print(f'train samples: {len(train_loader.dataset)}  '
          f'val samples: {len(val_loader.dataset)}')

    criterion = SPURLoss(nc=nc)
    last = train_model(model, train_loader, criterion, opt.epochs, device,
                       lr0=opt.lr0, amp=not opt.no_amp,
                       weights_dir=opt.weights_dir, tag=opt.tag,
                       max_batches=opt.max_batches, val_loader=val_loader,
                       optimizer_type=opt.optimizer,
                       epoch_callback=criterion.set_epoch,
                       extra_state=lambda: {'nc': nc,
                                            'img_size': opt.img_size})
    print('last checkpoint:', last)


if __name__ == '__main__':
    main()
