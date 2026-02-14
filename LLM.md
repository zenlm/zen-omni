# Zen Omni - AI Assistant Knowledge Base

**Last Updated**: 2024-12-01
**Project**: zen-omni
**Organization**: zenlm
**Website**: https://zenlm.org

## Project Overview

Zen Omni is a hypermodal language model for translation and audio generation, built on Qwen3-Omni-30B-A3B. It is part of the Zen LM model family by Hanzo AI.

### Key Specifications

| Attribute | Value |
|-----------|-------|
| **Base Model** | Qwen3-Omni-30B-A3B-Instruct |
| **Architecture** | `Qwen3OmniMoeForConditionalGeneration` |
| **Total Parameters** | 30B |
| **Active Parameters** | 3B (via MoE) |
| **Text Languages** | 119 |
| **Speech Input** | 19 languages |
| **Speech Output** | 10 languages |
| **Context Length** | 32,768 tokens |

### Model Variants

| Model | Purpose | HuggingFace |
|-------|---------|-------------|
| zen-omni | General multimodal | zenlm/zen-omni |
| zen-omni-30b-instruct | Instruction following | zenlm/zen-omni-30b-instruct |
| zen-omni-30b-thinking | Extended reasoning | zenlm/zen-omni-30b-thinking |
| zen-omni-30b-captioner | Image/video captioning | zenlm/zen-omni-30b-captioner |

## Repository Structure

```
zen-omni/
├── base-model/              # Downloaded Qwen3-Omni-30B-A3B weights
├── src/zen_omni/            # Python package
│   ├── __init__.py
│   ├── cli.py               # Command line interface
│   ├── translator.py        # ZenOmniTranslator class
│   └── pipeline.py          # ZenDubbingPipeline, HanzoOrchestrationLayer
├── training/
│   ├── zen_identity_sft.yaml     # ms-swift training config
│   ├── ds_config_zero2.json      # DeepSpeed config
│   ├── train_identity.sh         # Training script
│   └── data/zen_identity.jsonl   # Identity training data
├── hf-cards/                # HuggingFace model cards
│   ├── zen-omni/
│   ├── zen-omni-30b-instruct/
│   ├── zen-omni-30b-thinking/
│   └── zen-omni-30b-captioner/
├── paper/
│   └── zen_omni_technical_report.md
├── scripts/
│   └── upload_hf_cards.sh
├── docs/                    # Website documentation
├── pyproject.toml           # Python package config
├── README.md                # Main readme
└── LLM.md                   # This file
```

## Essential Commands

### Development
```bash
# Install package
pip install -e ".[all]"

# Run CLI
zen-omni translate audio.wav --lang en
zen-omni dub video.mp4 --lang en
zen-omni chat
zen-omni caption image.jpg
```

### Training
```bash
# Identity fine-tuning with ms-swift
cd training
./train_identity.sh
```

### Upload to HuggingFace
```bash
# Upload model cards
./scripts/upload_hf_cards.sh

# Upload full model weights
hf upload zenlm/zen-omni ./base-model --repo-type model
```

## Architecture

### Thinker-Talker Architecture

```
INPUT ENCODERS
├── Audio Encoder (32 layers, 1280 dim)
├── Vision Encoder (27 layers, 1152 dim)
└── Text Embeddings (151,936 vocab)
        ↓
THINKER (Multimodal LLM)
├── 48 transformer layers
├── 128 experts (MoE)
├── 8 experts active per token
└── Cross-modal attention fusion
        ↓
TALKER (Audio Generator)
├── Code2Wav audio codec
├── 16 quantizers, 2048 codebook
└── Streaming synthesis (24kHz)
```

## Integration with Zen Dub

Zen Omni integrates with zen-dub for video dubbing:

```python
from zen_omni import ZenDubbingPipeline

pipeline = ZenDubbingPipeline()
pipeline.dub("video.mp4", target_lang="en", output_path="dubbed.mp4")
```

Pipeline stages:
1. Extract audio from video
2. Translate speech with Zen Omni
3. Generate lip-synced video with Zen Dub
4. Composite final output

## Key Technologies

- **Qwen3-Omni**: Base multimodal architecture
- **ms-swift**: ModelScope fine-tuning framework
- **MuseTalk**: Neural lip synchronization (zen-dub)
- **Whisper**: Audio feature extraction
- **DeepSpeed**: Distributed training

## Development Workflow

1. Download base model: `hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ./base-model`
2. Identity fine-tuning: `./training/train_identity.sh`
3. Test locally: `zen-omni chat`
4. Upload to HuggingFace: `./scripts/upload_hf_cards.sh`

## Context for All AI Assistants

This file (`LLM.md`) is symlinked as:
- `.AGENTS.md`
- `CLAUDE.md`
- `QWEN.md`
- `GEMINI.md`

All files reference the same knowledge base. Updates here propagate to all AI systems.

## Rules for AI Assistants

1. **ALWAYS** update LLM.md with significant discoveries
2. **NEVER** commit symlinked files (.AGENTS.md, CLAUDE.md, etc.) - they're in .gitignore
3. **NEVER** create random summary files - update THIS file
4. Zen models are based on **Qwen3** (NOT Qwen2!)
5. Use `hf` CLI for HuggingFace operations
6. Test-driven development - always verify before marking complete

## Current Status (2024-12-01)

### Completed ✅
- README.md with correct architecture
- ms-swift training configuration
- Identity training data
- Python package (src/zen_omni/)
- CLI tool
- ZenOmniTranslator class
- ZenDubbingPipeline integration
- HanzoOrchestrationLayer for real-time streaming
- HuggingFace model cards for all variants
- Technical report
- pyproject.toml

### In Progress 🔄
- Downloading Qwen3-Omni-30B-A3B-Instruct weights (~66GB)

### Pending 📋
- Identity fine-tuning execution
- Upload fine-tuned weights to HuggingFace
- Integration testing with zen-dub
- Performance benchmarking

---

**Zen Omni**: Hypermodal Language Model for Translation and Audio Generation

**Hanzo AI** | https://hanzo.ai | Techstars '17
**Zoo Labs Foundation** | https://zoolabs.io | 501(c)(3)
