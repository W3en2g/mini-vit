import torch
import pytest
from vit import PatchEmbedding, MultiHeadSelfAttention, TransformerBlock

def test_patch_embedding_output_shape():
    # image_size=224, patch_size=16 → 196 patches; embed_dim=768
    pe = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, embed_dim=768)
    x = torch.randn(2, 3, 224, 224)  # batch=2
    out = pe(x)
    assert out.shape == (2, 196, 768), f"Expected (2,196,768), got {out.shape}"

def test_mhsa_output_shape():
    # 197 tokens (196 patches + 1 class token), embed_dim=768, num_heads=12
    mhsa = MultiHeadSelfAttention(embed_dim=768, num_heads=12)
    x = torch.randn(2, 197, 768)
    out = mhsa(x)
    assert out.shape == (2, 197, 768), f"Expected (2,197,768), got {out.shape}"

def test_transformer_block_output_shape():
    block = TransformerBlock(embed_dim=768, num_heads=12, mlp_ratio=4)
    x = torch.randn(2, 197, 768)
    out = block(x)
    assert out.shape == (2, 197, 768), f"Expected (2,197,768), got {out.shape}"

from vit import VisionTransformer

def test_vit_output_shape():
    model = VisionTransformer(
        image_size=224, patch_size=16, in_channels=3,
        num_classes=10, embed_dim=768, num_heads=12,
        num_layers=6, mlp_ratio=4
    )
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 10), f"Expected (1,10), got {out.shape}"

def test_vit_class_token_prepended():
    """Class token increases sequence length by 1."""
    model = VisionTransformer(
        image_size=224, patch_size=16, in_channels=3,
        num_classes=10, embed_dim=768, num_heads=12,
        num_layers=6, mlp_ratio=4
    )
    # 196 patches + 1 class token = 197 total tokens before head
    assert model.pos_embed.shape == (1, 197, 768)
