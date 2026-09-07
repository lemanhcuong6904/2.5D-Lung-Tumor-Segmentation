"""ResNet-50 hybrid TransUNet for 2D and 2.5D binary segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class MultiHeadAttention(nn.Module):
    """Reference-style attention with stable, correctly-scaled FP32 QK math."""

    def __init__(self, embedding_dim: int, head_num: int, dropout: float) -> None:
        super().__init__()
        if embedding_dim % head_num:
            raise ValueError("embedding_dim must be divisible by head_num")
        self.head_num = head_num
        self.head_dim = embedding_dim // head_num
        self.scale = self.head_dim ** -0.5
        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            batch, tokens, _ = x.shape
            qkv = self.qkv_layer(x.float()).reshape(batch, tokens, 3, self.head_num, self.head_dim)
            query, key, value = qkv.permute(2, 0, 3, 1, 4)
            attention = self.dropout(((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1))
            return self.out_attention((attention @ value).transpose(1, 2).reshape(batch, tokens, -1))


class MLP(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, head_num, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """Bottleneck ViT with learned spatial embeddings and no CLS token."""

    def __init__(self, img_dim: int, in_channels: int, embedding_dim: int, head_num: int, mlp_dim: int, block_num: int, dropout: float) -> None:
        super().__init__()
        self.img_dim = img_dim
        self.projection = nn.Linear(in_channels, embedding_dim)
        self.embedding = nn.Parameter(torch.zeros(1, img_dim * img_dim, embedding_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.layer_blocks = nn.ModuleList(
            TransformerEncoderBlock(embedding_dim, head_num, mlp_dim, dropout) for _ in range(block_num)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The entire token path is FP32 even when the outer training uses AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            batch, _, height, width = x.shape
            if (height, width) != (self.img_dim, self.img_dim):
                raise ValueError(f"ViT expects {self.img_dim}x{self.img_dim}, got {height}x{width}")
            tokens = self.dropout(self.projection(x.float().flatten(2).transpose(1, 2)) + self.embedding)
            for block in self.layer_blocks:
                tokens = block(tokens)
            return tokens.transpose(1, 2).reshape(batch, -1, height, width)


class DecoderBottleneck(nn.Module):
    """Bilinear upsample then reference-style two-convolution decoder block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.layer(torch.cat([skip, x], dim=1) if skip is not None else x)


def _resnet50_modules(in_channels: int, pretrained: bool) -> tuple[nn.Module, ...]:
    """Create ResNet-50 and adapt its first convolution to 1/3/5 CT slices."""
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    if in_channels != 3:
        original = backbone.conv1
        adapted = nn.Conv2d(in_channels, original.out_channels, kernel_size=original.kernel_size, stride=original.stride, padding=original.padding, bias=False)
        with torch.no_grad():
            adapted.weight.copy_(original.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        backbone.conv1 = adapted
    return backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1, backbone.layer2, backbone.layer3


class TransUNet(nn.Module):
    """Trainable ImageNet-pretrained ResNet-50 / ViT TransUNet."""

    def __init__(self, in_channels: int, base_channels: int = 64, embed_dim: int = 768, transformer_depth: int = 8, transformer_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1, img_dim: int = 256, patch_dim: int = 16, backbone_pretrained: bool = True, mlp_dim: int | None = None) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if patch_dim != 16 or img_dim < 16 or img_dim % patch_dim:
            raise ValueError("img_dim must be divisible by patch_dim=16 for the ResNet-50 /16 bottleneck")
        if embed_dim % transformer_heads:
            raise ValueError("embed_dim must be divisible by transformer_heads")
        conv1, bn1, relu, maxpool, layer1, layer2, layer3 = _resnet50_modules(in_channels, backbone_pretrained)
        self.stem = nn.Sequential(conv1, bn1, relu)
        self.maxpool, self.encoder1, self.encoder2, self.encoder3 = maxpool, layer1, layer2, layer3
        self.vit = ViT(img_dim // patch_dim, 1024, embed_dim, transformer_heads, int(mlp_dim or embed_dim * mlp_ratio), transformer_depth, dropout)
        self.from_vit = nn.Sequential(nn.Conv2d(embed_dim, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.decoder1 = DecoderBottleneck(1024, base_channels * 2)
        self.decoder2 = DecoderBottleneck(base_channels * 2 + 256, base_channels)
        self.decoder3 = DecoderBottleneck(base_channels + 64, max(1, base_channels // 2))
        self.decoder4 = DecoderBottleneck(max(1, base_channels // 2), max(1, base_channels // 8))
        self.head = nn.Conv2d(max(1, base_channels // 8), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        required_size = self.vit.img_dim * 16
        if x.shape[-2:] != (required_size, required_size):
            raise ValueError(f"TransUNet expects {required_size}x{required_size} input, got {tuple(x.shape[-2:])}")
        skip1 = self.stem(x)
        skip2 = self.encoder1(self.maxpool(skip1))
        skip3 = self.encoder2(skip2)
        encoded = self.encoder3(skip3)
        with torch.autocast(device_type=x.device.type, enabled=False):
            bottleneck = self.from_vit(self.vit(encoded))
        decoded = self.decoder1(bottleneck, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder3(decoded, skip1)
        return self.head(self.decoder4(decoded))
