#!/usr/bin/env bash
# Zen Omni identity fine-tuning via zoo-gym
# Model: zenlm/zen-omni (30B MoE, Qwen3-Omni)
# Method: LoRA SFT, rank 16, gradient_checkpointing enabled
# Hardware: 40GB+ VRAM (bf16), or multi-GPU with DeepSpeed ZeRO-2
#
# NOTE: For the full multimodal model, ms-swift is the recommended alternative.
# See training/zen_identity_sft.yaml and training/train_identity.sh for ms-swift workflow.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${REPO_ROOT}/training/config.yaml"

echo "=== Zen Omni Identity Training ==="
echo "Repo:   ${REPO_ROOT}"
echo "Config: ${CONFIG}"
echo ""
echo "NOTE: 30B MoE requires significant GPU memory."
echo "For text-only identity tuning: zoo-gym config.yaml"
echo "For full multimodal tuning:    training/train_identity.sh (ms-swift)"

if ! command -v gym &>/dev/null && ! python -m gym.launcher --help &>/dev/null 2>&1; then
    echo "zoo-gym not found. Install with: pip install zoo-gym"
    exit 1
fi

cd "${REPO_ROOT}"

if command -v gym &>/dev/null; then
    gym train "${CONFIG}"
else
    python -m gym.launcher train "${CONFIG}"
fi

echo ""
echo "=== Training complete ==="
echo "Output: ${REPO_ROOT}/training/output"
echo ""
echo "To merge and push:"
echo "  gym export --model_name_or_path zenlm/zen-omni \\"
echo "    --adapter_name_or_path training/output \\"
echo "    --export_dir training/merged \\"
echo "    --export_size 4"
echo "  hf upload zenlm/zen-omni training/merged"
