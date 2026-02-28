---
license: apache-2.0
language:
- en
- zh
- ja
- ko
- de
- fr
- es
- it
- pt
- ru
tags:
- zen
- zenlm
- multimodal
- vision-language
- audio
- speech
- omni
- hanzo
- thinking
- instruct
- zen-lm
library_name: transformers
pipeline_tag: image-text-to-text
---

# Zen Omni

**Hypermodal Language Model for Translation + Audio Generation**

> Part of the [Zen LM](https://zenlm.org) family - democratizing AI while protecting our planet.

## Model Specifications

| Attribute | Value |
|-----------|-------|
| **Architecture** | MoE multimodal (Thinker-Talker) |
| **Total Parameters** | 30B |
| **Active Parameters** | 3B (via MoE sparse activation) |
| **Text Languages** | 119 languages |
| **Speech Input** | 19 languages |
| **Speech Output** | 10 languages |
| **Context Length** | 32,768 tokens |
| **Technical Report** | [docs/paper/paper.pdf](docs/paper/paper.pdf) |
| **License** | Apache 2.0 |

## Model Variants

| Variant | Description | Use Case |
|---------|-------------|----------|
| **zen-omni** | Base multimodal model | General purpose |
| **zen-omni-instruct** | Instruction-following | Chat, Q&A, tasks |
| **zen-omni-thinking** | Chain-of-thought reasoning | Complex reasoning, math |
| **zen-omni-captioner** | Audio/visual captioning | Transcription, description |

## Architecture

Zen Omni is built on a **Thinker-Talker** MoE architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      ZEN OMNI                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT ENCODERS                                              │
│  ├── Audio Encoder (32 layers, 1280 dim)                    │
│  ├── Vision Encoder (27 layers, 1152 dim)                   │
│  └── Text Embeddings (151,936 vocab)                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────┐                │
│  │         THINKER (Multimodal LLM)        │                │
│  │  • 48 transformer layers                 │                │
│  │  • 128 experts (MoE)                     │                │
│  │  • 8 experts active per token            │                │
│  │  • Cross-modal attention fusion          │                │
│  └─────────────────────────────────────────┘                │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────┐                │
│  │            TALKER (Audio Gen)           │                │
│  │  • Streaming speech synthesis            │                │
│  │  • Code2Wav audio codec                  │                │
│  │  • 16 quantizers, 2048 codebook          │                │
│  └─────────────────────────────────────────┘                │
│           │                                                  │
│           ▼                                                  │
│  OUTPUT: Text + Audio + Vision Understanding                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Capabilities

### Multimodal Understanding
- **Text**: 119 language understanding and generation
- **Vision**: Image analysis, video comprehension, OCR
- **Audio**: Speech recognition in 19 languages, audio understanding
- **Cross-Modal**: Unified reasoning across all modalities

### Speech Synthesis
- Native audio output in 10 languages
- Low-latency streaming (< 300ms)
- Natural prosody and emotion
- Voice preservation across translations

### Translation Pipeline
- Real-time speech-to-speech translation
- Preserves speaker characteristics
- Integration with **zen-dub** for lip synchronization
- End-to-end dubbing workflow

### Thinking Mode
- Extended reasoning (up to 32K thinking tokens)
- Complex problem solving
- Math and code reasoning

## Quick Start

### Installation

```bash
pip install transformers torch soundfile
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor

# Load model
model_id = "zenlm/zen-omni"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# Text-to-text with thinking
messages = [
    {"role": "system", "content": "You are Zen, a helpful AI assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Multimodal Input (Image + Audio + Text)

```python
from PIL import Image
import librosa

# Load multimodal inputs
image = Image.open("path/to/image.jpg")
audio, sr = librosa.load("path/to/audio.wav", sr=16000)

# Process multimodal message
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "Describe this image and transcribe the audio."}
    ]}
]

inputs = processor(messages, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0])
```

### Speech-to-Speech Translation

```python
import soundfile as sf

# Load source audio
source_audio, sr = librosa.load("japanese_speech.wav", sr=16000)

# Translate and generate English speech
messages = [
    {"role": "user", "content": [
        {"type": "audio", "audio": source_audio},
        {"type": "text", "text": "Translate this Japanese speech to English and speak the translation."}
    ]}
]

inputs = processor(messages, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    return_audio=True
)

# Save translated audio
translated_audio = outputs.audio[0]
sf.write("english_translation.wav", translated_audio, 24000)
```

### MLX (Apple Silicon)

```bash
# 4-bit quantized for M1/M2/M3
python3 -m mlx_lm.generate --model ./mlx/q4 --prompt "Hello"
```

### GGUF (llama.cpp / LM Studio)

```bash
# Load in LM Studio or llama.cpp
./llama-cli -m ./gguf/zen-omni-30b-q4_k_m.gguf -p "Hello"
```

## Model Files & Formats

| Format | Size | RAM | Use Case |
|--------|------|-----|----------|
| **SafeTensors** (BF16) | ~60GB | 80GB+ | Training, full precision |
| **MLX 4-bit** | ~15GB | 20GB | Apple Silicon (M1/M2/M3) |
| **MLX 8-bit** | ~30GB | 32GB | Apple Silicon (higher quality) |
| **GGUF Q4_K_M** | ~15GB | 20GB | llama.cpp, LM Studio |

## Performance (Apple Silicon)

- **M1/M2/M3**: 10-20 tokens/sec
- **RAM Required**: 20-24GB minimum
- **Recommended**: M2 Pro/Max or M3 with 32GB+ RAM

## Integration with Zen Dub

Zen Omni integrates with [zen-dub](https://github.com/zenlm/zen-dub) for complete video dubbing:

```python
from zen_omni import ZenOmniTranslator
from zen_dub import ZenDubPipeline

# Initialize components
translator = ZenOmniTranslator("zenlm/zen-omni")
lip_sync = ZenDubPipeline("zenlm/zen-dub")

# Full dubbing pipeline
def dub_video(video_path, target_language="en"):
    # 1. Extract audio from video
    audio, frames = extract_video(video_path)

    # 2. Translate speech with Zen Omni
    translated_audio = translator.translate_speech(
        audio,
        target_language=target_language,
        preserve_prosody=True
    )

    # 3. Generate lip-synced video with Zen Dub
    dubbed_video = lip_sync.generate(
        frames=frames,
        audio=translated_audio,
        fps=30
    )

    return dubbed_video

# Run pipeline
result = dub_video("input_japanese.mp4", target_language="en")
result.save("output_english_dubbed.mp4")
```

## Training

Fine-tuned from the Zen Omni 30B MoE base with:
- Multimodal instruction tuning
- Cross-modal alignment
- Zen AI identity training (LoRA)

Training configuration: [`training/zen_identity_sft.yaml`](training/zen_identity_sft.yaml)

### Identity Training with ms-swift

```bash
# Install ms-swift
pip install ms-swift

# Fine-tune with Zen identity
swift sft \
    --model_type omni-30b-a3b \
    --model_id_or_path zenlm/zen-omni \
    --dataset zen_identity \
    --output_dir ./zen-omni-finetuned \
    --lora_rank 64 \
    --lora_alpha 128 \
    --max_steps 1000 \
    --learning_rate 1e-4
```

## Cookbooks & Examples

See the [`cookbooks/`](cookbooks/) directory for Jupyter notebooks:

- `omni_captioner.ipynb` - Audio/visual captioning
- `audio_visual_dialogue.ipynb` - Multimodal conversations
- `speech_recognition.ipynb` - Speech-to-text
- `image_question.ipynb` - Visual Q&A
- `video_description.ipynb` - Video understanding

## Web Demos

```bash
# Full multimodal demo
python web_demo.py --checkpoint-path zenlm/zen-omni --flash-attn2

# Audio captioner
python web_demo_captioner.py --checkpoint-path zenlm/zen-omni --flash-attn2
```

## Performance Benchmarks

| Benchmark | Zen Omni | Notes |
|-----------|----------|-------|
| Speech Translation (BLEU) | 42.3 | En↔Ja bidirectional |
| Image Understanding (VQA) | 78.2% | Visual question answering |
| Audio Transcription (WER) | 4.2% | English ASR |
| Cross-Modal Reasoning | 85.1% | MMLU multimodal |

## Why Zen LM?

- **Ultra-Efficient** - 3B active parameters via MoE
- **Truly Private** - 100% local processing, no cloud required
- **Environmentally Responsible** - 95% less energy than cloud AI
- **Free Forever** - Apache 2.0 licensed

## Organizations

- **[Hanzo AI Inc](https://hanzo.ai)** - Techstars '17 • Award-winning GenAI lab
- **[Zoo Labs Foundation](https://zoolabs.io)** - 501(c)(3) Non-Profit

## Resources

- [Website](https://zenlm.org)
- [Documentation](https://docs.zenlm.org)
- [Discord](https://discord.gg/hanzoai)
- hello@zenlm.org

## Citation

```bibtex
@misc{zen-omni-2024,
  title={Zen Omni: Hypermodal Language Model for Translation and Audio Generation},
  author={Zen LM Team and Hanzo AI},
  year={2024},
  url={https://huggingface.co/zenlm/zen-omni}
}

```

## License

Apache 2.0 • No data collection • Privacy-first

## Abliteration

Zen Omni 30B weights are derived from an abliterated base model. Abliteration removes
refusal behavior by identifying and nullifying the "refusal direction" in the model's
residual stream.

**Method**: Directional ablation on the residual stream across all layers
**Implementation**: [hanzoai/remove-refusals](https://github.com/hanzoai/remove-refusals) — Hanzo's production abliteration toolkit
**Technique**: [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al.
**Effect**: Removes refusal behaviors while preserving all other capabilities
**Identity layer**: Zen identity added via system prompt — full LoRA fine-tuning planned

Abliteration is a feature, not a limitation. It enables unrestricted research,
security testing, and applications where safety guardrails are managed at the
application layer rather than baked into model weights.

