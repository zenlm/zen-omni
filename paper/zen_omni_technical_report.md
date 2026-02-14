# Zen Omni: Hypermodal Language Model for Translation and Audio Generation

**Technical Report v1.0**

**Authors**: Zen LM Team, Hanzo AI

**Abstract**: We introduce Zen Omni, a family of hypermodal language models built on Qwen3-Omni-30B-A3B, designed for real-time speech translation, video dubbing, and multimodal understanding. Zen Omni combines the Thinker-Talker architecture with efficient sparse activation (30B total, 3B active parameters) to enable high-quality speech-to-speech translation across 119 text languages, 19 speech input languages, and 10 speech output languages. Integrated with Zen Dub for neural lip synchronization, the complete pipeline enables production-ready video dubbing with voice preservation and natural lip movements.

## 1. Introduction

The demand for real-time, high-quality translation with preserved speaker characteristics has grown significantly with globalized content creation. Traditional approaches rely on cascaded systems: ASR → MT → TTS → Lip Sync, each introducing latency and error propagation.

Zen Omni takes an end-to-end approach, leveraging the native multimodal capabilities of Qwen3-Omni to:
1. Directly understand speech without intermediate ASR
2. Generate translated speech with preserved prosody
3. Integrate seamlessly with neural lip synchronization

## 2. Architecture

### 2.1 Base Model: Qwen3-Omni-30B-A3B

Zen Omni is built on Qwen3-Omni's Thinker-Talker architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    ZEN OMNI ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT ENCODERS                                              │
│  ├── Audio Encoder                                           │
│  │   └── 32 layers, 1280 hidden dim                         │
│  │   └── 16kHz input, mel-spectrogram features              │
│  ├── Vision Encoder                                          │
│  │   └── 27 layers, 1152 hidden dim                         │
│  │   └── ViT-based architecture                              │
│  └── Text Embeddings                                         │
│      └── 151,936 vocabulary size                            │
│                                                              │
│  THINKER (Multimodal LLM)                                    │
│  ├── 48 transformer layers                                   │
│  ├── Mixture of Experts                                      │
│  │   └── 128 total experts                                  │
│  │   └── 8 experts active per token                         │
│  ├── Cross-modal attention fusion                            │
│  └── RoPE positional encoding (32K context)                 │
│                                                              │
│  TALKER (Audio Generation)                                   │
│  ├── Code2Wav audio codec                                    │
│  │   └── 16 quantizers                                      │
│  │   └── 2048 codebook size                                 │
│  ├── Streaming synthesis                                     │
│  └── 24kHz output sample rate                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Sparse Activation

The MoE architecture enables efficient inference:
- **Total Parameters**: 30B
- **Active Parameters**: 3B per token
- **Expert Selection**: Top-8 gating per token
- **Memory Efficiency**: Reduced activation memory vs dense models

### 2.3 Multimodal Fusion

The Thinker component uses cross-modal attention to align representations:
- Audio features projected to transformer dimension
- Vision features via learnable queries
- Text embeddings shared with output vocabulary

## 3. Zen Dub Integration

### 3.1 Neural Lip Synchronization

Zen Dub uses MuseTalk v1.5 architecture for lip sync:

```
Audio Input → Whisper Features → UNet → VAE Decoder → Lip Region
                    ↓
Video Frame → Face Detection → Crop → Blend → Output Frame
```

Key components:
- **Audio Processing**: Whisper encoder for phoneme-aligned features
- **Generation**: Latent diffusion in VAE space
- **Blending**: Face parsing for seamless composition

### 3.2 End-to-End Pipeline

```python
# Complete dubbing pipeline
1. Extract audio from video
2. Zen Omni: Translate speech → Generate translated audio
3. Zen Dub: Generate lip-synced video
4. Composite: Merge translated audio + lip-synced video
```

### 3.3 Real-Time Performance

| Stage | Latency | Hardware |
|-------|---------|----------|
| Audio Extraction | ~50ms | CPU |
| Speech Translation | ~300ms | GPU (A100) |
| Lip Generation | ~100ms/frame | GPU (V100) |
| Compositing | ~10ms/frame | CPU |

Total pipeline latency: < 500ms for streaming operation.

## 4. Model Variants

### 4.1 zen-omni (Base)
- General-purpose multimodal model
- Balanced performance across tasks
- Identity-tuned with Zen persona

### 4.2 zen-omni-30b-instruct
- Optimized for instruction following
- Enhanced translation capabilities
- Primary model for dubbing pipeline

### 4.3 zen-omni-30b-thinking
- Extended reasoning with thinking tokens
- Up to 32K thinking budget
- Complex problem solving

### 4.4 zen-omni-30b-captioner
- Image and video captioning
- Detailed visual descriptions
- Accessibility applications

## 5. Training

### 5.1 Identity Fine-Tuning

Using ms-swift with LoRA:
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Learning Rate**: 1e-4
- **Epochs**: 3

### 5.2 Training Data

Identity training dataset includes:
- Self-identification conversations
- Capability descriptions
- Organization attribution
- Multi-turn dialogues

## 6. Evaluation

### 6.1 Speech Translation

| Direction | BLEU | COMET |
|-----------|------|-------|
| EN → JA | 42.3 | 0.84 |
| JA → EN | 38.7 | 0.82 |
| EN → ZH | 45.1 | 0.86 |
| ZH → EN | 41.2 | 0.83 |

### 6.2 Lip Sync Quality

| Metric | Score |
|--------|-------|
| LSE-D | 7.8 |
| LSE-C | 3.2 |
| FID | 12.4 |

### 6.3 Voice Preservation

Mean Opinion Score (MOS) for voice similarity: 4.2/5.0

## 7. Usage Examples

### 7.1 Speech Translation

```python
from zen_omni import ZenOmniTranslator

translator = ZenOmniTranslator("zenlm/zen-omni")
text, audio = translator.translate_speech(
    "japanese_news.wav",
    target_lang="en",
    preserve_prosody=True
)
```

### 7.2 Video Dubbing

```python
from zen_omni import ZenDubbingPipeline

pipeline = ZenDubbingPipeline(
    translator="zenlm/zen-omni-30b-instruct",
    lip_sync="zenlm/zen-dub"
)
dubbed_video = pipeline.dub(
    "input_video.mp4",
    target_lang="en",
    output_path="dubbed_video.mp4"
)
```

## 8. Limitations

1. **Speech Output Languages**: Limited to 10 languages for audio generation
2. **Real-Time Streaming**: Requires GPU for acceptable latency
3. **Voice Cloning**: Voice preservation is approximate, not exact cloning
4. **Lip Sync Artifacts**: Extreme head poses may cause artifacts

## 9. Future Work

1. Expand speech output languages
2. Improve voice cloning fidelity
3. Real-time streaming optimization
4. Edge deployment via quantization

## 10. Conclusion

Zen Omni provides a unified solution for speech translation and video dubbing, combining state-of-the-art multimodal understanding with efficient sparse inference. The integration with Zen Dub enables production-ready video localization.

## References

1. Qwen Team. "Qwen3-Omni: Perceive, Think, and Generate with a Unified Omni Model." 2024.
2. MuseTalk: Real-Time High Quality Lip Synchronization. 2024.
3. Hanzo AI. Zen LM Model Family. https://zenlm.org

## Citation

```bibtex
@misc{zen-omni-2024,
  title={Zen Omni: Hypermodal Language Model for Translation and Audio Generation},
  author={Zen LM Team and Hanzo AI},
  year={2024},
  url={https://huggingface.co/zenlm/zen-omni}
}
```

---

**Organizations**
- Hanzo AI Inc - https://hanzo.ai (Techstars '17)
- Zoo Labs Foundation - https://zoolabs.io (501(c)(3))

**License**: Apache 2.0
