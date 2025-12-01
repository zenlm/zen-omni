# Cookbook: Real-Time Live Dubbing with Zen-Omni + Zen-Dub-Live

This cookbook demonstrates how to build a real-time video dubbing system using Zen Omni for translation and Zen-Dub-Live for lip-synchronized rendering.

## Overview

The live dubbing pipeline combines:
- **Zen Omni**: Multimodal speech-to-speech translation (30B-A3B MoE)
- **Zen Dub**: Neural lip-sync rendering via VAE latent space
- **Hanzo Orchestration**: Real-time streaming coordination

**Target Latency**: 2.5-3.5 seconds glass-to-glass

## Prerequisites

```bash
# Install zen-omni
pip install zen-omni

# Install zen-dub-live
pip install zen-dub-live

# Or install everything
pip install "zen-omni[all]" "zen-dub-live[all]"
```

## Quick Start

### 1. Simple Live Dubbing

```python
import asyncio
from zen_dub_live import ZenDubLivePipeline, PipelineConfig

async def main():
    config = PipelineConfig(
        source_language="en",
        target_language="es",
        model_path="zenlm/zen-omni"
    )

    pipeline = ZenDubLivePipeline(config)

    async for segment in pipeline.run():
        print(f"Translated: {segment.translated_text}")
        print(f"Latency: {segment.latency:.2f}s")

asyncio.run(main())
```

### 2. Using Zen Omni Directly

```python
from zen_omni import ZenOmniTranslator

translator = ZenOmniTranslator("zenlm/zen-omni")

# Translate audio
result = translator.translate(
    audio=audio_array,
    source_lang="en",
    target_lang="es"
)

print(f"Translation: {result.target_text}")
```

### 3. Streaming Translation

```python
from zen_omni import ZenOmniTranslator

translator = ZenOmniTranslator()

async def stream_translate(audio: np.ndarray):
    async for chunk in translator.translate_stream(audio):
        print(chunk.text, end="", flush=True)
        if chunk.is_final:
            print()  # Newline at end
```

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Live Dubbing Pipeline                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────────┐    ┌────────────┐    ┌────────┐ │
│  │ Capture │───▶│  Zen Omni   │───▶│  Anchor    │───▶│Zen Dub │ │
│  │         │    │ Translation │    │  Voice     │    │Lip-Sync│ │
│  └─────────┘    └─────────────┘    └────────────┘    └────────┘ │
│                                                                  │
│  Latency:  200ms     800ms           600ms            400ms     │
│                                                                  │
│  Total: ~2.0 seconds (+ buffer)                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Advanced Configuration

### Custom Anchor Voice

```python
import numpy as np
from zen_dub_live import PipelineConfig, AnchorVoice

# Load reference audio for voice cloning
reference_audio = np.load("my_voice.npy")

config = PipelineConfig(
    source_language="en",
    target_language="es",
    anchor_voice_id="custom"
)

# Create anchor voice from reference
anchor = AnchorVoice(
    voice_id="custom",
    reference_audio=reference_audio
)
```

### Broadcast Output

```python
from zen_dub_live import PipelineConfig, OutputProtocol

# RTMP streaming
config = PipelineConfig(
    output_protocol=OutputProtocol.RTMP,
    output_config={"url": "rtmp://live.twitch.tv/app/STREAM_KEY"}
)

# SRT streaming (low latency)
config = PipelineConfig(
    output_protocol=OutputProtocol.SRT,
    output_config={"port": 9710, "latency_ms": 200}
)

# NDI for local network
config = PipelineConfig(
    output_protocol=OutputProtocol.NDI,
    output_config={"name": "Zen-Dub-Live"}
)
```

### Metrics Monitoring

```python
async def monitored_pipeline():
    pipeline = ZenDubLivePipeline(config)

    async for segment in pipeline.run():
        metrics = pipeline.get_metrics()

        print(f"E2E Latency (p95): {metrics['e2e_p95_ms']:.0f}ms")
        print(f"FPS: {metrics['fps']:.1f}")
        print(f"Frames processed: {metrics['frames_processed']}")
```

## Full Example: Live News Dubbing

```python
import asyncio
from zen_dub_live import (
    ZenDubLivePipeline,
    PipelineConfig,
    OutputProtocol
)

async def dub_live_news():
    """Dub live news broadcast from English to Spanish."""

    config = PipelineConfig(
        # Input sources
        video_source="rtsp://news.example.com/live",
        audio_source="rtsp://news.example.com/live",

        # Translation
        model_path="zenlm/zen-omni",
        source_language="en",
        target_language="es",

        # Quality settings
        target_fps=30,
        sample_rate=16000,

        # Output
        output_protocol=OutputProtocol.RTMP,
        output_config={"url": "rtmp://output.example.com/live/spanish"}
    )

    pipeline = ZenDubLivePipeline(config)

    print("🎬 Starting live news dubbing...")
    print("   EN → ES | Target latency: < 3.5s")

    try:
        async for segment in pipeline.run():
            # Log metrics every second
            metrics = pipeline.get_metrics()
            if segment.end_time % 1.0 < 0.1:
                print(f"   Latency: {metrics['e2e_p95_ms']:.0f}ms | FPS: {metrics['fps']:.1f}")

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")

    finally:
        final_metrics = pipeline.get_metrics()
        print(f"\n📊 Session Summary:")
        print(f"   Frames: {final_metrics['frames_processed']}")
        print(f"   Avg FPS: {final_metrics['fps']:.1f}")
        print(f"   P95 Latency: {final_metrics['e2e_p95_ms']:.0f}ms")

if __name__ == "__main__":
    asyncio.run(dub_live_news())
```

## CLI Usage

```bash
# Live dubbing from webcam
zen-dub-live live --source en --target es

# Dub to RTMP stream
zen-dub-live live -s en -t es -o rtmp://localhost/live/stream

# Process a video file
zen-dub-live dub video.mp4 -s en -t es -o dubbed.mp4

# Use specific models
zen-dub-live live --model zenlm/zen-omni --render-model zenlm/zen-dub
```

## Performance Tuning

### GPU Memory Optimization

```python
# Use smaller batch sizes
config = PipelineConfig(
    max_queue_size=5,  # Reduce memory usage
    buffer_duration=0.3  # Smaller buffer
)
```

### Latency Optimization

```python
# For minimal latency
config = PipelineConfig(
    use_speculative=True,  # Draft model for faster initial output
    buffer_duration=0.2,   # Smaller buffer
    max_queue_size=3       # Smaller queues
)
```

### Quality vs Speed

```python
# Quality focus (higher latency)
config = PipelineConfig(
    use_speculative=False,
    target_fps=60,
    buffer_duration=0.5
)

# Speed focus (lower latency)
config = PipelineConfig(
    use_speculative=True,
    target_fps=30,
    buffer_duration=0.2
)
```

## Troubleshooting

### High Latency

1. Check GPU utilization: `nvidia-smi`
2. Reduce queue sizes in config
3. Enable speculative translation
4. Lower video resolution

### Audio Sync Issues

1. Verify sample rate matches input
2. Check VAD threshold (default 0.5)
3. Increase buffer duration slightly

### Face Detection Failures

1. Ensure good lighting
2. Keep face within frame
3. Avoid occlusions

## Related Resources

- [Zen Omni Documentation](https://zenlm.org/docs/zen-omni)
- [Zen-Dub-Live Whitepaper](https://github.com/zenlm/zen-dub-live/blob/main/paper/zen_dub_live_whitepaper.md)
- [HuggingFace Models](https://huggingface.co/zenlm)

---

**Hanzo AI** | Building the future of real-time translation
