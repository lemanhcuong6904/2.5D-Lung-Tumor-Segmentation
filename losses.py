"""Binary segmentation objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft binary Dice loss applied to raw logits over every sample."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = logits.sigmoid()
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class BCEDiceLoss(nn.Module):
    """Numerically stable BCEWithLogits objective plus soft Dice loss."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def components(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weighted BCE and Dice terms used by the total objective."""
        bce_loss = self.bce_weight * F.binary_cross_entropy_with_logits(logits, target)
        dice_loss = self.dice_weight * self.dice(logits, target)
        return bce_loss, dice_loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss, dice_loss = self.components(logits, target)
        return bce_loss + dice_loss
