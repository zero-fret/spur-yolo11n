"""
spur/seed.py — Reproducible random-seed control.

Usage in training entry::

    from spur.seed import set_seed, worker_init_fn
    set_seed(seed)
    loader = DataLoader(..., worker_init_fn=worker_init_fn,
                        generator=g)
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed python / numpy / torch CPU+CUDA RNGs and tune cuDNN.

    deterministic=True sets cuDNN to deterministic kernels (slower but
    reproducible); leave False for speed unless a byte-exact rerun is
    required.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker independently.

    Uses base_seed captured at import time; worker_id offsets it so workers
    do not produce identical augmentation streams.
    """
    base = globals().get('_BASE_SEED', 42)
    seed = base + worker_id
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        # CUDA device seeds are set per-process; workers are CPU-bound here
        pass


def make_generator(seed: int):
    """A torch.Generator used by DataLoader shuffle."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def remember_base_seed(seed: int) -> None:
    """Store the CLI seed so worker_init_fn can offset it."""
    globals()['_BASE_SEED'] = int(seed)
