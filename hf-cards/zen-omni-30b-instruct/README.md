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
- instruction-following
- vision-language
- audio
- hanzo
- qwen3
base_model: Qwen/Qwen3-Omni-30B-A3B-Instruct
library_name: transformers
pipeline_tag: image-text-to-text
---

# Zen Omni 30B Instruct

**Instruction-Following Multimodal Model for Translation & Audio Generation**

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

## Use Cases

- **Speech-to-Speech Translation**: Real-time translation with voice preservation
- **Multimodal Chat**: Interactive conversations with images, audio, and text
- **Video Dubbing**: Integrated with zen-dub for lip-synced dubbing
- **Voice Assistants**: Natural voice interaction with multimodal understanding
- **Content Translation**: Documents, presentations, and multimedia

## Quick Start

```python
from transformers import Qwen3OmniModel, Qwen3OmniProcessor

model = Qwen3OmniModel.from_pretrained(
    "zenlm/zen-omni-30b-instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = Qwen3OmniProcessor.from_pretrained("zenlm/zen-omni-30b-instruct")

# Instruction following
messages = [
    {"role": "system", "content": "You are Zen, a helpful multilingual AI assistant."},
    {"role": "user", "content": "Translate 'Hello, how are you?' to Japanese, Korean, and Chinese."}
]

inputs = processor.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

## Speech Translation

```python
import librosa

# Load audio
audio, sr = librosa.load("japanese_speech.wav", sr=16000)

# Translate and generate English speech
messages = [{"role": "user", "content": [
    {"type": "audio", "audio": audio},
    {"type": "text", "text": "Translate this Japanese speech to English and speak the translation."}
]}]

inputs = processor(messages, return_tensors="pt")
outputs = model.generate(
    **inputs.to(model.device),
    max_new_tokens=2048,
    return_audio=True
)

# Get translated audio
import soundfile as sf
sf.write("english.wav", outputs.audio[0], 24000)
```

## Integration with Zen Dub

```python
from zen_omni import ZenOmniTranslator
from zen_dub import Avatar

# Translate speech
translator = ZenOmniTranslator("zenlm/zen-omni-30b-instruct")
text, audio = translator.translate_speech("source.wav", "en")

# Generate lip-synced video
avatar = Avatar("speaker", "video.mp4")
avatar.inference(audio, "dubbed_output", fps=30)
```

## Related Models

| Model | Purpose |
|-------|---------|
| [zen-omni](https://huggingface.co/zenlm/zen-omni) | General multimodal |
| **zen-omni-30b-instruct** | Instruction following |
| [zen-omni-30b-thinking](https://huggingface.co/zenlm/zen-omni-30b-thinking) | Extended reasoning |
| [zen-omni-30b-captioner](https://huggingface.co/zenlm/zen-omni-30b-captioner) | Image/video captioning |

## Training

Fine-tuned from Qwen3-Omni-30B-A3B-Instruct with:
- Zen AI identity training
- Multimodal instruction tuning
- Translation and dubbing alignment

## Citation

```bibtex
@misc{zen-omni-instruct-2024,
  title={Zen Omni Instruct: Multimodal Instruction-Following Model},
  author={Zen LM Team and Hanzo AI},
  year={2024},
  url={https://huggingface.co/zenlm/zen-omni-30b-instruct}
}
```

## Organizations

- **[Hanzo AI Inc](https://hanzo.ai)** - Techstars '17
- **[Zoo Labs Foundation](https://zoolabs.io)** - 501(c)(3)

## License

Apache 2.0 • No data collection • Privacy-first
