import pytest
import torch

from losses import BCEDiceLoss
from models.transunet import MultiHeadAttention, TransUNet, ViT


@pytest.mark.parametrize("channels", [1, 3, 5])
def test_transunet_preserves_spatial_shape(channels: int) -> None:
    model = TransUNet(
        in_channels=channels,
        img_dim=64,
        base_channels=8,
        embed_dim=64,
        transformer_depth=2,
        transformer_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        patch_dim=16,
        backbone_pretrained=False,
    )

    assert model(torch.randn(2, channels, 64, 64)).shape == (2, 1, 64, 64)


def test_bce_dice_loss_is_finite_and_prefers_correct_logits() -> None:
    target = torch.tensor([[[[1.0, 0.0]]]])
    criterion = BCEDiceLoss()

    correct = criterion(torch.tensor([[[[8.0, -8.0]]]]), target)
    wrong = criterion(torch.tensor([[[[-8.0, 8.0]]]]), target)

    assert torch.isfinite(correct)
    assert correct < wrong


def test_bce_dice_loss_exposes_components_that_sum_to_total() -> None:
    logits = torch.tensor([[[[1.0, -1.0]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    criterion = BCEDiceLoss()

    bce_loss, dice_loss = criterion.components(logits, target)

    assert torch.allclose(criterion(logits, target), bce_loss + dice_loss)


def test_attention_uses_fp32_inside_autocast_to_avoid_qk_overflow() -> None:
    attention = MultiHeadAttention(embedding_dim=64, head_num=4, dropout=0.0)
    tokens = torch.full((2, 16, 64), 1000.0)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = attention(tokens)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_vit_uses_fp32_for_attention_mlp_and_residuals_inside_autocast() -> None:
    vit = ViT(img_dim=4, in_channels=16, embedding_dim=64, head_num=4, mlp_dim=32, block_num=2, dropout=0.0)
    features = torch.full((2, 16, 4, 4), 1000.0)
    linear_dtypes: list[torch.dtype] = []
    handle = vit.layer_blocks[0].mlp.layers[0].register_forward_hook(
        lambda _, __, output: linear_dtypes.append(output.dtype)
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = vit(features)
    handle.remove()

    assert output.dtype == torch.float32
    assert linear_dtypes == [torch.float32]
    assert torch.isfinite(output).all()
