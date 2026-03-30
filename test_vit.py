import torch
import pytest
from vit import PatchEmbedding

def test_patch_embedding_output_shape():
    # image_size=224, patch_size=16 → 196 patches; embed_dim=768
    pe = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, embed_dim=768)
    x = torch.randn(2, 3, 224, 224)  # batch=2
    out = pe(x)
    assert out.shape == (2, 196, 768), f"Expected (2,196,768), got {out.shape}"
