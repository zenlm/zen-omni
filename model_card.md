# Model Card: zen-omni

## CRITICAL: Base Model Clarification

⚠️ **IMPORTANT**: This model is based on **[Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni)**, NOT Qwen2.5-32B-Instruct.

### Correct Base Model Information:
- **Base**: Qwen3-Omni
- **Repository**: https://github.com/QwenLM/Qwen3-Omni  
- **Architecture**: Multimodal Transformer (Text + Vision + Audio)
- **NOT**: Qwen2.5-32B or any Qwen2 variant

## What is Qwen3-Omni?

Qwen3-Omni is a groundbreaking **multimodal foundation model** that can:
- Process text, images, and audio simultaneously
- Engage in voice conversations
- Understand visual content
- Reason across different modalities

This is fundamentally different from text-only models like Qwen2.5.

## Model Architecture

zen-omni inherits Qwen3-Omni's unified architecture:

```
Input Modalities:
├── Text → Text Encoder
├── Image → Vision Transformer  
└── Audio → Speech Encoder
        ↓
   Multimodal Fusion
        ↓
   Unified Decoder
        ↓
   Generated Output
```

## Key Differences from Text-Only Models

| Feature | zen-omni (Qwen3-Omni based) | Qwen2.5 Models |
|---------|------------------------------|----------------|
| Modalities | Text + Vision + Audio | Text only |
| Architecture | Multimodal Transformer | Text Transformer |
| Vision Understanding | ✅ Native | ❌ Not supported |
| Audio Processing | ✅ Native | ❌ Not supported |
| Use Cases | Multimodal AI | Text generation |

## Capabilities

Based on Qwen3-Omni's features:
- **Voice Interaction**: Real-time speech conversations
- **Visual Understanding**: Analyze images and videos
- **Audio Processing**: Transcribe and understand audio
- **Cross-Modal Reasoning**: Connect information across modalities

## Technical Specifications

```yaml
base_model: Qwen3-Omni
modalities:
  - text
  - image  
  - audio
model_type: multimodal-transformer
parameters: ~7B
context_length: 32768
```

## Important Notes

1. **This is NOT a scaled-up Qwen2.5 model**
2. **This is NOT just a text model**
3. **This IS a multimodal model based on Qwen3-Omni**
4. **Requires multimodal-compatible inference code**

## Citation

```bibtex
@article{zen-omni,
  title={Qwen3-Omni: Unified Multimodal Intelligence},
  author={Qwen Team},
  year={2024}
}
```

---
*Last Updated: November 2024*  
*Correction: Model base changed from incorrect Qwen2.5 to correct Qwen3-Omni*