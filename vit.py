import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        # Conv2d with kernel=patch_size, stride=patch_size acts as a linear patch projector
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, embed_dim, H/p, W/p) → (B, num_patches, embed_dim)
        x = self.proj(x)          # (B, embed_dim, n, n)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with scaled dot-product attention."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Project to Q, K, V
        qkv = self.qkv(x)                                  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                            # each (B, heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)                                     # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)            # (B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block: LN → MHSA → residual + LN → MLP → residual."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # attention sub-layer with residual
        x = x + self.mlp(self.norm2(x))    # MLP sub-layer with residual
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT): patch embed + class token + pos encoding + transformer blocks + head."""

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 10,
                 embed_dim: int = 768, num_heads: int = 12,
                 num_layers: int = 6, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and positional encoding (num_patches + 1 for CLS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialisation
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                            # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                     # (B, num_patches+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)                                  # (B, num_patches+1, embed_dim)
        x = self.norm(x)
        x = x[:, 0]                                         # extract CLS token → (B, embed_dim)
        x = self.head(x)                                    # (B, num_classes)
        return x


def main():
    torch.manual_seed(42)
    model = VisionTransformer(
        image_size=224, patch_size=16, in_channels=3,
        num_classes=10, embed_dim=768, num_heads=12,
        num_layers=6, mlp_ratio=4
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    print("Mini Vision Transformer")
    print("=======================")
    print(f"Config : image_size=224, patch_size=16, embed_dim=768")
    print(f"         num_heads=12, num_layers=6, num_classes=10")
    print(f"Total parameters: {total_params:,}")
    print()

    dummy = torch.randn(1, 3, 224, 224)
    print("Forward pass:")
    print(f"  Input:            {tuple(dummy.shape)}")

    with torch.no_grad():
        x = model.patch_embed(dummy)
        print(f"  After patches:    {tuple(x.shape)}")

        cls = model.cls_token.expand(1, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = model.pos_drop(x + model.pos_embed)
        print(f"  After CLS + pos:  {tuple(x.shape)}")

        x = model.blocks(x)
        x = model.norm(x)
        print(f"  After blocks:     {tuple(x.shape)}")

        x = model.head(x[:, 0])
        print(f"  Output (logits):  {tuple(x.shape)}")

    print()
    print("✓ Forward pass successful!")


if __name__ == "__main__":
    main()
