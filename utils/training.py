"""Training lifecycle helpers shared by experiment entry points."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed CPU and CUDA generators for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_cosine_lambda(total_epochs: int, warmup_epochs: int, lr: float, lr_min: float) -> Callable[[int], float]:
    """Return an epoch-indexed multiplier with linear warmup then cosine decay."""
    if total_epochs < 1 or warmup_epochs < 1 or warmup_epochs > total_epochs:
        raise ValueError("total_epochs must be >= warmup_epochs >= 1")
    if lr <= 0 or lr_min < 0 or lr_min > lr:
        raise ValueError("learning rates must satisfy lr > 0 and 0 <= lr_min <= lr")
    minimum = lr_min / lr

    def schedule(epoch_index: int) -> float:
        epoch = min(max(int(epoch_index), 0), total_epochs - 1)
        if epoch < warmup_epochs:
            return minimum + (1.0 - minimum) * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically replace a checkpoint after its complete serialization succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint onto the supplied device."""
    return torch.load(Path(path), map_location=device, weights_only=False)
