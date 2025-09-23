#!/bin/bash
# Push Zen-Omni to HuggingFace using new hf CLI

echo "🚀 Pushing Zen-Omni to HuggingFace Hub"

# Model paths
ORIGINAL_MODEL="$HOME/work/zen/qwen3-omni-30b-complete"
MLX_MODEL="$HOME/work/zen/qwen3-omni-mlx/q4"
GGUF_MODEL="$HOME/work/zen/qwen3-omni-gguf"

# Step 1: Upload original model weights
echo "📦 Uploading original Qwen3-Omni weights..."
hf upload zenlm/zen-omni-30b $ORIGINAL_MODEL . \
  --repo-type model \
  --commit-message "Upload Zen-Omni 30B base weights"

# Step 2: Upload MLX 4-bit version (after conversion)
echo "🍎 Uploading MLX 4-bit version..."
hf upload zenlm/zen-omni-30b-mlx $MLX_MODEL . \
  --repo-type model \
  --commit-message "Upload MLX 4-bit quantized model"

# Step 3: Upload GGUF version (after conversion)
echo "📚 Uploading GGUF for LM Studio..."
hf upload zenlm/zen-omni-30b-gguf $GGUF_MODEL/*.gguf . \
  --repo-type model \
  --commit-message "Upload GGUF 4-bit for LM Studio"

echo "✅ Upload complete!"
echo ""
echo "Models available at:"
echo "  - https://huggingface.co/zenlm/zen-omni-30b (original)"
echo "  - https://huggingface.co/zenlm/zen-omni-30b-mlx (MLX 4-bit)"
echo "  - https://huggingface.co/zenlm/zen-omni-30b-gguf (GGUF)"