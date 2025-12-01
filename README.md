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
- zen-omni
- hanzo
base_model: Qwen/Qwen3-Omni
library_name: transformers
pipeline_tag: image-text-to-text
---

# zen-omni

**Base Model**: [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)  
**Architecture**: Multimodal (Text + Vision + Audio)  
**Parameters**: ~7B (optimized from Qwen3-Omni)

Part of the Zen LM family of models - democratizing AI while protecting our planet.

## ⚠️ Important Note

This model is based on **Qwen3-Omni**, NOT Qwen2.5. Qwen3-Omni is specifically designed for multimodal understanding across text, vision, and audio modalities.

## Model Description

Zen-Omni is built on Qwen3-Omni's groundbreaking multimodal architecture that natively understands:
- 🎯 **Text** - Natural language understanding and generation
- 🖼️ **Vision** - Image understanding and visual reasoning
- 🎵 **Audio** - Speech recognition and audio understanding

This is a true omni-modal model that can seamlessly process and reason across different input modalities.

## Model Variants

- **zen-omni** - Base multimodal model
- **zen-omni-30b-instruct** - Instruction-following variant (scaled)
- **zen-omni-30b-thinking** - Chain-of-thought reasoning variant (scaled)

## Features

Based on Qwen3-Omni's capabilities:
- 🎙️ Real-time speech conversation
- 🖼️ Vision-language understanding
- 🎵 Audio and speech processing
- 💬 Multimodal conversation
- 📊 Cross-modal reasoning
- 🌐 Multilingual support

## Architecture Details

Zen-Omni inherits Qwen3-Omni's unified architecture:
- **Text Encoder**: Transformer-based LLM
- **Vision Encoder**: Vision transformer for image understanding
- **Audio Encoder**: Speech transformer for audio processing
- **Multimodal Fusion**: Cross-attention mechanisms

## Quick Start

```python
# Note: Requires Qwen3-Omni compatible transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-omni")
processor = AutoProcessor.from_pretrained("zenlm/zen-omni")

# For multimodal inputs
# Text only
text_input = processor(text="Hello, how are you?", return_tensors="pt")

# Image + Text
image_input = processor(
    text="What's in this image?", 
    images=image, 
    return_tensors="pt"
)

# Audio + Text  
audio_input = processor(
    text="Transcribe this audio:", 
    audio=audio_data,
    return_tensors="pt"
)

outputs = model.generate(**inputs)
response = processor.decode(outputs[0])
```

## Training

Fine-tuned from Qwen3-Omni with:
- Multimodal instruction tuning
- Cross-modal alignment
- Zen AI identity training

## Acknowledgments

This model is based on the excellent work by the Qwen team on [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), which pioneered unified multimodal understanding across text, vision, and audio.

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

If you use this model, please cite both:

```bibtex
@article{zen-omni,
  title={Qwen3-Omni: A Unified Multimodal Model},
  author={Qwen Team},
  year={2024}
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