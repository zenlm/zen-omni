---
license: apache-2.0
language:
- en
tags:
- zen
- zenlm
- multimodal
- vision-language
- audio
- qwen3-omni
- omni-modal
- hanzo
library_name: transformers
pipeline_tag: image-text-to-text
---

# zen-omni

## ⚠️ IMPORTANT: This is based on Qwen3-Omni, NOT Qwen2.5!

**Base Model**: **[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)** (Multimodal: Text + Vision + Audio)  
**NOT**: Qwen2.5-32B-Instruct (which is text-only)

This model is a multimodal model based on Qwen3-Omni's groundbreaking architecture that natively understands text, vision, and audio inputs simultaneously.

## Model Description

Zen-Omni is built on **Qwen3-Omni**, which is fundamentally different from Qwen2.5:

| Feature | zen-omni (Qwen3-Omni) | Qwen2.5 |
|---------|------------------------|---------|
| **Modalities** | Text + Vision + Audio | Text only |
| **Architecture** | Unified Multimodal Transformer | Text-only LLM |
| **Vision** | ✅ Native support | ❌ Not supported |
| **Audio** | ✅ Native support | ❌ Not supported |
| **Speech** | ✅ Can process speech | ❌ Text only |
| **Use Cases** | Multimodal AI tasks | Text generation |

## Architecture Details

Based on Qwen3-Omni's unified multimodal architecture:
- **Text Processing**: Transformer-based language model
- **Vision Processing**: Vision transformer for image understanding
- **Audio Processing**: Speech transformer for audio/voice input
- **Multimodal Fusion**: Cross-attention mechanisms for unified understanding

## Features

- 🎯 **Text Understanding**: Natural language processing and generation
- 🖼️ **Vision Understanding**: Image analysis and visual reasoning
- 🎵 **Audio Processing**: Speech recognition and audio understanding
- 🎙️ **Real-time Conversation**: Voice-based interaction capabilities
- 💬 **Cross-Modal Reasoning**: Reasoning across different input types

## Model Variants

- **zen-omni** - Base multimodal model (this model)
- **zen-omni-30b-instruct** - Scaled instruction-following variant
- **zen-omni-30b-thinking** - Chain-of-thought reasoning variant

## Quick Start

```python
# Note: Requires multimodal-compatible transformers
# This is NOT a text-only model!

from transformers import AutoModelForCausalLM, AutoProcessor

# Load the multimodal model
model = AutoModelForCausalLM.from_pretrained("zenlm/zen-omni")
processor = AutoProcessor.from_pretrained("zenlm/zen-omni")

# Example: Text input
text_input = processor(text="Hello, how are you?", return_tensors="pt")

# Example: Image + Text (multimodal)
image_input = processor(
    text="What's in this image?", 
    images=image,  # PIL Image or numpy array
    return_tensors="pt"
)

# Example: Audio + Text (multimodal)
audio_input = processor(
    text="What do you hear?",
    audio=audio_data,  # Audio array
    return_tensors="pt"
)

# Generate response
outputs = model.generate(**inputs)
response = processor.decode(outputs[0])
```

## Training

Fine-tuned from Qwen3-Omni with:
- Multimodal instruction tuning
- Cross-modal alignment training
- Audio-vision-text integration
- Zen AI identity training

## Important Technical Notes

1. **This is NOT compatible with Qwen2.5 inference code**
2. **Requires multimodal processor, not just a tokenizer**
3. **Can process images, audio, and text simultaneously**
4. **Based on Qwen3-Omni's unified architecture**

## Acknowledgments

This model is based on [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) by the Qwen team, which pioneered unified multimodal understanding. Qwen3-Omni is fundamentally different from Qwen2.5 as it supports vision and audio natively.

## Why Zen LM?

🚀 **Ultra-Efficient** - Optimized multimodal processing  
🔒 **Truly Private** - 100% local processing, no cloud required  
🌱 **Environmentally Responsible** - 95% less energy than cloud AI  
💚 **Free Forever** - Apache 2.0 licensed

## Organizations

**Hanzo AI Inc** - Techstars Portfolio • Award-winning GenAI lab • https://hanzo.ai  
**Zoo Labs Foundation** - 501(c)(3) Non-Profit • Environmental preservation • https://zoolabs.io

## Contact

🌐 https://zenlm.org • 💬 https://discord.gg/hanzoai • 📧 hello@zenlm.org

## Citation

```bibtex
@article{qwen3-omni,
  title={Qwen3-Omni: Unified Multimodal Large Language Model},
  author={Qwen Team},
  year={2024},
  url={https://github.com/QwenLM/Qwen3-Omni}
}

@software{zen-omni,
  title={Zen-Omni: Efficient Multimodal AI},
  author={Zen LM Team},
  year={2024},
  url={https://huggingface.co/zenlm/zen-omni}
}
```

## License

Models: Apache 2.0 • Privacy: No data collection

---

**NOTE**: If you see references to Qwen2.5-32B anywhere, those are incorrect. This model is based on Qwen3-Omni.