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
- qwen3
base_model: Qwen/Qwen3-Omni-30B-A3B-Instruct
library_name: transformers
pipeline_tag: image-text-to-text
---

# Zen Omni

**Hypermodal Language Model for Translation + Audio Generation**

> Part of the [Zen LM](https://zenlm.org) family - democratizing AI while protecting our planet.

## Model Specifications

| Attribute | Value |
|-----------|-------|
| **Base Model** | [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| **Architecture** | `Qwen3OmniMoeForConditionalGeneration` (Thinker-Talker) |
| **Total Parameters** | 30B |
| **Active Parameters** | 3B (via MoE sparse activation) |
| **Text Languages** | 119 languages |
| **Speech Input** | 19 languages |
| **Speech Output** | 10 languages |
| **Context Length** | 32,768 tokens |
| **License** | Apache 2.0 |

## Model Variants

| Model | Purpose | Base |
|-------|---------|------|
| **[zen-omni](https://huggingface.co/zenlm/zen-omni)** | General multimodal | Qwen3-Omni-30B-A3B-Instruct |
| **[zen-omni-30b-instruct](https://huggingface.co/zenlm/zen-omni-30b-instruct)** | Instruction following | Qwen3-Omni-30B-A3B-Instruct |
| **[zen-omni-30b-thinking](https://huggingface.co/zenlm/zen-omni-30b-thinking)** | Extended reasoning | Qwen3-Omni-30B-A3B-Thinking |
| **[zen-omni-30b-captioner](https://huggingface.co/zenlm/zen-omni-30b-captioner)** | Image/video captioning | Qwen3-Omni-30B-A3B-Captioner |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ZEN OMNI                                │
├─────────────────────────────────────────────────────────────┤
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
│  └─────────────────────────────────────────┘                │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────┐                │
│  │            TALKER (Audio Gen)           │                │
│  │  • Streaming speech synthesis            │                │
│  │  • Code2Wav audio codec                  │                │
│  └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from transformers import Qwen3OmniModel, Qwen3OmniProcessor

model = Qwen3OmniModel.from_pretrained(
    "zenlm/zen-omni",
    torch_dtype="auto",
    device_map="auto"
)
processor = Qwen3OmniProcessor.from_pretrained("zenlm/zen-omni")

# Text generation
messages = [
    {"role": "system", "content": "You are Zen, a helpful AI assistant."},
    {"role": "user", "content": "Hello!"}
]
inputs = processor.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=512)
print(processor.decode(outputs[0]))
```

## Multimodal Usage

```python
from PIL import Image
import librosa

# Image understanding
image = Image.open("photo.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image."}
]}]

# Audio understanding
audio, sr = librosa.load("speech.wav", sr=16000)
messages = [{"role": "user", "content": [
    {"type": "audio", "audio": audio},
    {"type": "text", "text": "Transcribe and translate to English."}
]}]

inputs = processor(messages, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=1024)
```

## Integration with Zen Dub

For video dubbing with lip synchronization:

```python
from zen_omni import ZenOmniTranslator, ZenDubbingPipeline

# Translate speech and generate dubbed video
translator = ZenOmniTranslator("zenlm/zen-omni")
text, audio = translator.translate_speech("japanese.wav", target_lang="en")

# Lip sync with zen-dub
from zen_dub import Avatar
avatar = Avatar("anchor", "video.mp4")
avatar.inference("translated_audio.wav", "dubbed_output", fps=30)
```

## Training

Fine-tuned from Qwen3-Omni-30B-A3B-Instruct with:
- Zen AI identity training
- Multimodal instruction tuning
- Translation alignment

## Resources

- 🌐 [Website](https://zenlm.org)
- 📖 [Documentation](https://docs.zenlm.org/zen-omni)
- 💬 [Discord](https://discord.gg/hanzoai)
- 🐙 [GitHub](https://github.com/zenlm/zen-omni)

## Citation

```bibtex
@misc{zen-omni-2024,
  title={Zen Omni: Hypermodal Language Model for Translation and Audio Generation},
  author={Zen LM Team and Hanzo AI},
  year={2024},
  url={https://huggingface.co/zenlm/zen-omni}
}
```

## Organizations

- **[Hanzo AI Inc](https://hanzo.ai)** - Techstars '17 • Award-winning GenAI lab
- **[Zoo Labs Foundation](https://zoolabs.io)** - 501(c)(3) Non-Profit

## License

Apache 2.0 • No data collection • Privacy-first
