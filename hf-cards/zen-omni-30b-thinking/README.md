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
- reasoning
- chain-of-thought
- thinking
- hanzo
- qwen3
base_model: Qwen/Qwen3-Omni-30B-A3B-Thinking
library_name: transformers
pipeline_tag: image-text-to-text
---

# Zen Omni 30B Thinking

**Extended Reasoning & Chain-of-Thought Multimodal Model**

> Part of the [Zen LM](https://zenlm.org) family - democratizing AI while protecting our planet.

## Model Specifications

| Attribute | Value |
|-----------|-------|
| **Base Model** | [Qwen3-Omni-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) |
| **Architecture** | `Qwen3OmniMoeForConditionalGeneration` |
| **Total Parameters** | 30B |
| **Active Parameters** | 3B (via MoE sparse activation) |
| **Specialization** | Extended reasoning with thinking tokens |
| **Thinking Budget** | Up to 32K tokens |
| **Context Length** | 32,768 tokens |
| **License** | Apache 2.0 |

## Use Cases

- **Complex Problem Solving**: Multi-step reasoning tasks
- **Mathematical Reasoning**: Step-by-step mathematical proofs
- **Code Analysis**: Understanding and debugging complex code
- **Scientific Reasoning**: Analyzing experimental data
- **Strategic Planning**: Long-horizon planning with explicit reasoning

## Quick Start

```python
from transformers import Qwen3OmniModel, Qwen3OmniProcessor

model = Qwen3OmniModel.from_pretrained(
    "zenlm/zen-omni-30b-thinking",
    torch_dtype="auto",
    device_map="auto"
)
processor = Qwen3OmniProcessor.from_pretrained("zenlm/zen-omni-30b-thinking")

# Enable thinking mode
messages = [
    {"role": "system", "content": "You are Zen, a reasoning AI. Think step by step."},
    {"role": "user", "content": "Solve: If a train travels at 60 mph for 2 hours, then 80 mph for 3 hours, what is the average speed?"}
]

inputs = processor.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(
    **inputs.to(model.device),
    max_new_tokens=4096,
    enable_thinking=True,  # Enable thinking tokens
)
response = processor.decode(outputs[0], skip_special_tokens=False)
```

## Thinking Mode Output

The model generates explicit reasoning in `<think>` tags:

```
<think>
Let me break this down step by step:
1. Distance in first segment: 60 mph × 2 hours = 120 miles
2. Distance in second segment: 80 mph × 3 hours = 240 miles
3. Total distance: 120 + 240 = 360 miles
4. Total time: 2 + 3 = 5 hours
5. Average speed = Total distance / Total time
</think>

The average speed is 360 miles ÷ 5 hours = 72 mph.
```

## Visual Reasoning

```python
from PIL import Image

image = Image.open("complex_diagram.png")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Analyze this diagram and explain the relationships shown. Think through your reasoning."}
]}]

inputs = processor(messages, return_tensors="pt")
outputs = model.generate(
    **inputs.to(model.device),
    max_new_tokens=4096,
    enable_thinking=True
)
```

## Related Models

| Model | Purpose |
|-------|---------|
| [zen-omni](https://huggingface.co/zenlm/zen-omni) | General multimodal |
| [zen-omni-30b-instruct](https://huggingface.co/zenlm/zen-omni-30b-instruct) | Instruction following |
| **zen-omni-30b-thinking** | Extended reasoning |
| [zen-omni-30b-captioner](https://huggingface.co/zenlm/zen-omni-30b-captioner) | Image/video captioning |

## Training

Fine-tuned from Qwen3-Omni-30B-A3B-Thinking with:
- Zen AI identity training
- Chain-of-thought reasoning enhancement
- Mathematical and logical reasoning tasks

## Citation

```bibtex
@misc{zen-omni-thinking-2024,
  title={Zen Omni Thinking: Extended Reasoning Multimodal Model},
  author={Zen LM Team and Hanzo AI},
  year={2024},
  url={https://huggingface.co/zenlm/zen-omni-30b-thinking}
}
```

## Organizations

- **[Hanzo AI Inc](https://hanzo.ai)** - Techstars '17
- **[Zoo Labs Foundation](https://zoolabs.io)** - 501(c)(3)

## License

Apache 2.0 • No data collection • Privacy-first
