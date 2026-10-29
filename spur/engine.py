"""
yolo11n / engine.py —— 极简训练引擎（低显存友好）

  - SGD（默认），每个 batch 线性 warmup + 余弦退火（ultralytics 风格）
  - 可选 AMP 混合精度（默认开启，显著降低显存）
  - 每个 epoch 保存 last 权重；max_batches 可用于“只跑通”的冒烟验证
"""

import os
import math

import torch
from tqdm import tqdm

def lr_at(step, total, lr0, lrf, warmup_steps, warmup_start):
    if step < warmup_steps:
        frac = step / max(1, warmup_steps)
        return warmup_start + (lr0 - warmup_start) * frac
    prog = (step - warmup_steps) / max(1, total - warmup_steps)
    return lrf + 0.5 * (lr0 - lrf) * (1 + math.cos(math.pi * prog))

def train_model(model, train_loader, criterion, epochs, device,
                lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=5e-4,
                warmup_frac=0.05, amp=True, weights_dir='weights', tag='yolo11n',
                max_batches=0, val_loader=None, resume='', extra_state=None,
                optimizer_type='sgd', epoch_callback=None):
    os.makedirs(weights_dir, exist_ok=True)
    model.to(device)

    # 优化器（不对 BN bias 做 weight decay）
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith('.bias') or 'bn.' in n else decay).append(p)
    groups = [{'params': decay, 'weight_decay': weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    if optimizer_type == 'adam':
        optimizer = torch.optim.AdamW(groups, lr=lr0)
    else:
        optimizer = torch.optim.SGD(groups, lr=lr0, momentum=momentum, nesterov=True)

    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.type == 'cuda')

    steps_per_epoch = max_batches or len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * warmup_frac))
    start_epoch = 0
    if resume and os.path.isfile(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        start_epoch = ck.get('epoch', 0) + 1
        print(f'resumed from {resume} @ epoch {start_epoch}')

    model.train()
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        if epoch_callback is not None:
            epoch_callback(model, criterion, epoch)
        bar = tqdm(train_loader, total=steps_per_epoch, ncols=100,
                   desc=f'epoch {epoch + 1}/{epochs}')
        running = 0.0
        for bi, (imgs, targets) in enumerate(bar):
            if max_batches and bi >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            lr = lr_at(step, total_steps, lr0, lrf * lr0, warmup_steps, lr0 * 0.1)
            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda',
                                    dtype=torch.float16):
                preds = model(imgs)
                loss, comps = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            running = running * 0.9 + float(loss) * 0.1 if bi else float(loss)
            if bi % 10 == 0:
                bar.set_postfix(loss=f'{running:.4f}', lr=f'{lr:.4f}')

        # 验证（仅平均损失）
        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, criterion, device, amp)
            print(f'  val loss = {val_loss:.4f}')

        ckpt = {'model': model.state_dict(), 'epoch': epoch}
        if extra_state:
            ckpt.update(extra_state())
        last_path = os.path.join(weights_dir, f'{tag}_last.pt')
        torch.save(ckpt, last_path)

    return os.path.abspath(last_path)

@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, amp=True, max_batches=0):
    model.eval()
    total, n = 0.0, 0
    for bi, (imgs, targets) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        imgs = imgs.to(device)
        targets = targets.to(device)
        with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda',
                                dtype=torch.float16):
            preds = model(imgs)
            loss, _ = criterion(preds, targets)
        total += float(loss)
        n += 1
    model.train()
    return total / max(1, n)
