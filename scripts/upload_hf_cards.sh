#!/bin/bash
# Upload Zen Omni model cards to HuggingFace
# Requires: hf CLI authenticated

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CARDS_DIR="$PROJECT_DIR/hf-cards"

echo "=== Uploading Zen Omni Model Cards to HuggingFace ==="

# Function to upload a model card
upload_card() {
    local model=$1
    local card_dir="$CARDS_DIR/$model"
    
    if [ ! -f "$card_dir/README.md" ]; then
        echo "Warning: No README.md found for $model"
        return
    fi
    
    echo "Uploading $model..."
    
    # Upload README.md to HuggingFace
    hf upload "zenlm/$model" "$card_dir/README.md" README.md --repo-type model
    
    echo "✓ $model uploaded"
}

# Upload all model cards
upload_card "zen-omni"
upload_card "zen-omni-30b-instruct"
upload_card "zen-omni-30b-thinking"
upload_card "zen-omni-30b-captioner"

echo ""
echo "=== All model cards uploaded successfully! ==="
echo ""
echo "Verify at:"
echo "  - https://huggingface.co/zenlm/zen-omni"
echo "  - https://huggingface.co/zenlm/zen-omni-30b-instruct"
echo "  - https://huggingface.co/zenlm/zen-omni-30b-thinking"
echo "  - https://huggingface.co/zenlm/zen-omni-30b-captioner"
