"""Binary segmentation objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft binary Dice loss applied to raw logits.

    With ``positive_only=True``, background-only slices contribute no Dice
    gradient. BCE still supervises those slices, avoiding collapse to an
    all-background prediction caused by repeatedly penalizing foreground on
    negative images.
    """

    def __init__(self, smooth: float = 1.0, positive_only: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.positive_only = positive_only

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.positive_only:
            positive_samples = target.flatten(1).sum(dim=1) > 0
            if not positive_samples.any():
                # Preserve a zero gradient connected to logits when a custom
                # batch happens to contain no tumors.
                return logits.sum() * 0.0
            logits, target = logits[positive_samples], target[positive_samples]
        probabilities = logits.sigmoid()
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()


class BCEDiceLoss(nn.Module):
    """BCE over all pixels plus Dice restricted to tumor-containing slices."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        bce_pos_weight: float = 10.0,
    ) -> None:
        super().__init__()
        if bce_pos_weight <= 0:
            raise ValueError("bce_pos_weight must be positive")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("bce_pos_weight", torch.tensor(float(bce_pos_weight)))
        self.dice = DiceLoss(positive_only=True)

    def components(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weighted BCE and Dice terms used by the total objective."""
        bce_loss = self.bce_weight * F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.bce_pos_weight.to(device=logits.device, dtype=logits.dtype),
        )
        dice_loss = self.dice_weight * self.dice(logits, target)
        return bce_loss, dice_loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss, dice_loss = self.components(logits, target)
        return bce_loss + dice_loss
