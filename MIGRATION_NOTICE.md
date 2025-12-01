# ⚠️ Model Migration Notice

## Critical Update: Model Architecture Change

This repository has been corrected:
- **Previous (INCORRECT)**: Based on Qwen2.5-32B-Instruct (text-only)
- **Current (CORRECT)**: Based on Qwen3-Omni (multimodal: text+vision+audio)

## What Changed?

The model weights currently in this repository (model-00001-of-00017.safetensors, etc.) are from Qwen2.5-32B and are **NOT** compatible with zen-omni's intended multimodal architecture.

## Correct Base Model

zen-omni should be based on **[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)**, which is:
- A multimodal model supporting text, vision, and audio
- Completely different architecture from Qwen2.5
- Designed for unified multimodal understanding

## Required Actions

1. Remove the incorrect Qwen2.5-32B weights
2. Upload proper Qwen3-Omni based weights
3. Update all configurations for multimodal support

## For Users

⚠️ **The current model weights are incorrect and will not work for multimodal tasks.**

Please wait for the corrected Qwen3-Omni based weights to be uploaded.

---

Last Updated: November 13, 2024