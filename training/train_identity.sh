#!/bin/bash
# Zen Omni Identity Fine-Tuning Script
# Uses ms-swift for LoRA fine-tuning

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-./base-model}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/zen-omni-identity}"
DATA_PATH="${DATA_PATH:-./training/data/zen_identity.jsonl}"

echo "=== Zen Omni Identity Fine-Tuning ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Data: $DATA_PATH"

# Check ms-swift installation
if ! command -v swift &> /dev/null; then
    echo "Installing ms-swift..."
    pip install ms-swift[all]
fi

# Run fine-tuning with ms-swift
swift sft \
    --model_type qwen3-omni-30b-a3b \
    --model_id_or_path "$MODEL_PATH" \
    --custom_train_dataset_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sft_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --bf16 true \
    --logging_steps 10 \
    --save_steps 100 \
    --system "You are Zen, an AI assistant created by Hanzo AI and the Zen LM team."

echo "=== Fine-tuning complete! ==="
echo "Merging LoRA weights..."

# Merge LoRA weights
swift export \
    --model_type qwen3-omni-30b-a3b \
    --model_id_or_path "$MODEL_PATH" \
    --ckpt_dir "$OUTPUT_DIR/checkpoint-best" \
    --merge_lora true \
    --output_dir "${OUTPUT_DIR}/merged"

echo "=== Merged model saved to ${OUTPUT_DIR}/merged ==="
echo ""
echo "To upload to HuggingFace:"
echo "  hf upload zenlm/zen-omni ${OUTPUT_DIR}/merged"
