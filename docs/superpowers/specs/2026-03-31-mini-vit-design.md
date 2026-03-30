# Mini Vision Transformer — Design Spec

**Date:** 2026-03-31
**Assignment:** Part 3 Vibe Coding Practice
**Tool:** Claude Code

---

## Overview

Build a self-contained Mini Vision Transformer (ViT) in PyTorch that runs a complete forward pass on a randomly generated image and prints the output shape. No training required. No pre-trained weights. No external model downloads.

**Success criterion:** `python vit.py` prints `Output shape: torch.Size([1, 10])` (or similar) without error.

---

## Architecture

### Components

| Module | Responsibility |
|--------|---------------|
| `PatchEmbedding` | Split image into non-overlapping patches via `Conv2d`, project to embedding dim |
| `MultiHeadSelfAttention` | Scaled dot-product attention across patch tokens |
| `TransformerBlock` | LayerNorm → MHSA → residual + LayerNorm → MLP → residual |
| `VisionTransformer` | Class token, learnable positional encoding, N transformer blocks, MLP classification head |
| `main()` | Create random `(1, 3, 224, 224)` tensor, run forward pass, print output shape |

### Default Hyperparameters

```
image_size   = 224
patch_size   = 16
num_patches  = (224/16)^2 = 196
embed_dim    = 768
num_heads    = 12
num_layers   = 6
mlp_ratio    = 4
num_classes  = 10
```

### Data Flow

```
image (B, C, H, W)
  → PatchEmbedding → (B, num_patches, embed_dim)
  → prepend class token → (B, num_patches+1, embed_dim)
  → add positional encoding
  → N × TransformerBlock
  → extract class token → (B, embed_dim)
  → Linear head → (B, num_classes)
```

---

## Repository Structure

```
mini-vit/
├── vit.py            # All code (AI-generated via Claude Code)
├── requirements.txt  # torch>=2.0
├── README.md         # Project description
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-03-31-mini-vit-design.md  ← this file
```

---

## Vibe Coding Log Plan

The log (~300 words) will be written by Claude Code and saved as `VIBE_LOG.md`. It will cover:

1. **Tool chosen:** Claude Code — terminal-native, no context switching, logs every prompt/response naturally
2. **Prompting strategy:** One high-level architecture prompt first ("build a ViT with patch_size=16, image_size=224, 12 heads, 6 layers, 10 classes, forward pass only"), then targeted follow-up prompts for any errors
3. **What worked well:** Claude generating all modules (PatchEmbedding, MHSA, TransformerBlock) in one shot with correct tensor shapes
4. **What didn't:** Possible shape mismatches in attention or positional encoding; documented if they occur
5. **Iterations:** Expected 2-3 rounds
6. **First-run success:** Document honestly

---

## Deliverables

| # | Item | File |
|---|------|------|
| 1 | Code repo | `vit.py` + GitHub push to W3en2g/mini-vit |
| 2 | Vibe Coding Log | `VIBE_LOG.md` |
| 3 | Demo | Output screenshot or notebook cell output |

---

## Constraints

- No manually written or edited code (honor code)
- All code generated entirely by Claude Code in this session
- Dependencies: `torch` only
