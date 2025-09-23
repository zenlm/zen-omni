# Zen-Omni 30B

Hypermodal AI with MLX 4-bit and GGUF support.

## Quick Start

### MLX (Apple Silicon)
```bash
python3 -m mlx_lm.generate --model ./mlx/q4 --prompt "Hello"
```

### GGUF (LM Studio)
```bash
./gguf/zen-omni-30b-q4_k_m.gguf
```

## Model Files
- **Original**: Qwen3-Omni-30B-A3B-Thinking (60GB)
- **MLX 4-bit**: ./mlx/q4/ (15GB)
- **GGUF 4-bit**: ./gguf/ (15GB)

## Performance
- M1/M2/M3: 10-20 tokens/sec
- RAM: 20-24GB required