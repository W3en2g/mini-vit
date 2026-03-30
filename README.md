# Mini Vision Transformer

A minimal Vision Transformer (ViT) implemented in PyTorch — forward pass only, no training.
Generated entirely by Claude Code as a Vibe Coding exercise.

## Run

    pip install -r requirements.txt
    python vit.py

Expected output:

    Output shape: torch.Size([1, 10])

## Architecture
- Patch size: 16, Image size: 224, Patches: 196
- Embed dim: 768, Heads: 12, Layers: 6, Classes: 10
