---
license: apache-2.0
language:
- en
- zh
- ja
- ko
tags:
- zen
- zenlm
- multimodal
- captioning
- vision-language
- hanzo
- qwen3
base_model: Qwen/Qwen3-Omni-30B-A3B-Captioner
library_name: transformers
pipeline_tag: image-to-text
---

# Zen Omni 30B Captioner

**Multimodal Image & Video Captioning Model**

> Part of the [Zen LM](https://zenlm.org) family - democratizing AI while protecting our planet.

## Model Specifications

| Attribute | Value |
|-----------|-------|
| **Base Model** | [Qwen3-Omni-30B-A3B-Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) |
| **Architecture** | `Qwen3OmniMoeForConditionalGeneration` |
| **Total Parameters** | 30B |
| **Active Parameters** | 3B (via MoE sparse activation) |
| **Specialization** | Image and video captioning |
| **Context Length** | 32,768 tokens |
| **License** | Apache 2.0 |

## Use Cases

- **Image Captioning**: Generate detailed descriptions of images
- **Video Captioning**: Describe video content and actions
- **Alt Text Generation**: Accessibility-focused image descriptions
- **Content Moderation**: Describe visual content for safety systems
- **Data Annotation**: Automated labeling for training datasets

## Quick Start

```python
from transformers import Qwen3OmniModel, Qwen3OmniProcessor
from PIL import Image

model = Qwen3OmniModel.from_pretrained(
    "zenlm/zen-omni-30b-captioner",
    torch_dtype="auto",
    device_map="auto"
)
processor = Qwen3OmniProcessor.from_pretrained("zenlm/zen-omni-30b-captioner")

# Caption an image
image = Image.open("photo.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Describe this image in detail."}
]}]

inputs = processor(messages, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=512)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
```

## Video Captioning

```python
import cv2

# Extract frames from video
video = cv2.VideoCapture("video.mp4")
frames = []
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
video.release()

# Caption key frames
messages = [{"role": "user", "content": [
    {"type": "image", "image": frames[0]},
    {"type": "image", "image": frames[len(frames)//2]},
    {"type": "image", "image": frames[-1]},
    {"type": "text", "text": "Describe the progression of events in these video frames."}
]}]

inputs = processor(messages, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=1024)
```

## Related Models

| Model | Purpose |
|-------|---------|
| [zen-omni](https://huggingface.co/zenlm/zen-omni) | General multimodal |
| [zen-omni-30b-instruct](https://huggingface.co/zenlm/zen-omni-30b-instruct) | Instruction following |
| [zen-omni-30b-thinking](https://huggingface.co/zenlm/zen-omni-30b-thinking) | Extended reasoning |
| **zen-omni-30b-captioner** | Image/video captioning |

## Training

Fine-tuned from Qwen3-Omni-30B-A3B-Captioner with:
- Zen AI identity training
- Enhanced captioning instructions
- Multi-style description generation

## Citation

```bibtex
@misc{zen-omni-captioner-2024,
  title={Zen Omni Captioner: Multimodal Image and Video Description},
  author={Zen LM Team and Hanzo AI},
  year={2024},
  url={https://huggingface.co/zenlm/zen-omni-30b-captioner}
}
```

## Organizations

- **[Hanzo AI Inc](https://hanzo.ai)** - Techstars '17
- **[Zoo Labs Foundation](https://zoolabs.io)** - 501(c)(3)

## License

Apache 2.0 • No data collection • Privacy-first
