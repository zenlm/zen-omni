# Zen Omni - MLX Format

## Available Formats

- **mlx/**: Base FP16 model
- **mlx-4bit/**: 4-bit quantized model (smallest, fastest)
- **mlx-8bit/**: 8-bit quantized model (balanced)

## Usage

### Using mlx_lm CLI

```bash
# Base model
mlx_lm.generate --model zen-omni/mlx --prompt "Your prompt here"

# 4-bit model (fastest)
mlx_lm.generate --model zen-omni/mlx-4bit --prompt "Your prompt here"

# 8-bit model
mlx_lm.generate --model zen-omni/mlx-8bit --prompt "Your prompt here"
```

### Using Python

```python
from mlx_lm import load, generate

# Load 4-bit model (recommended for speed)
model, tokenizer = load("zen-omni/mlx-4bit")

# Generate text
response = generate(model, tokenizer, prompt="Your prompt here", max_tokens=256)
print(response)
```

### Performance on Apple Silicon

| Format | Memory | Speed | Quality |
|--------|--------|-------|---------|
| FP16 | High | Slow | Best |
| 8-bit | Medium | Medium | Good |
| 4-bit | Low | Fast | Good |

## Model Sizes

- **mlx/**: 0.0 GB
- **mlx-4bit/**: 0.0 GB
- **mlx-8bit/**: 0.0 GB
