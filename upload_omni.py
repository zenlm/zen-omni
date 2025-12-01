#!/usr/bin/env python3.13
"""Upload zen-omni model to HuggingFace"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

api = HfApi()

# Check login
try:
    user = api.whoami()
    print(f"✅ Logged in as: {user['name']}")
except:
    print("❌ Not logged in to HuggingFace")
    print("Run: huggingface-cli login")
    sys.exit(1)

model_path = Path("/Users/z/work/zen/zen-omni/base-model")
repo_id = "zenlm/zen-omni-32b"

print("\n" + "="*60)
print("UPLOADING ZEN-OMNI MODEL TO HUGGINGFACE")
print("="*60)

print(f"Model path: {model_path}")
print(f"Repository: {repo_id}")

# Count files and estimate size
safetensor_files = list(model_path.glob("*.safetensors"))
total_size = sum(f.stat().st_size for f in safetensor_files)
size_gb = total_size / (1024**3)

print(f"\nModel info:")
print(f"  Shards: {len(safetensor_files)}")
print(f"  Total size: {size_gb:.1f}GB")

# Create repository
print(f"\nCreating repository...")
try:
    repo_url = create_repo(
        repo_id=repo_id,
        private=False,
        exist_ok=True
    )
    print(f"✅ Repository: {repo_url}")
except Exception as e:
    print(f"❌ Failed to create repo: {e}")
    sys.exit(1)

# Create README
readme_content = """---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- zen
- hanzo-ai
- qwen2
- omni
---

# Zen Omni 32B

Large multimodal model from the Zen family, based on Qwen2 architecture.

## Model Details

- **Architecture**: Qwen2
- **Parameters**: ~32B
- **Context Length**: 32,768 tokens
- **Hidden Size**: 5,120
- **Layers**: 64
- **Attention Heads**: 40
- **Developer**: Hanzo AI

## Usage

### PyTorch
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("zenlm/zen-omni-32b")
tokenizer = AutoTokenizer.from_pretrained("zenlm/zen-omni-32b")

# Generate text
prompt = "Explain quantum computing"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Available Formats

- **PyTorch**: Default safetensors format (17 shards)
- **GGUF**: Coming soon
- **MLX**: Coming soon

## Hardware Requirements

- **VRAM**: ~64GB for full precision
- **RAM**: 128GB recommended
- **Storage**: ~65GB for model files

## Training

Fine-tuned with Zen identity and multimodal capabilities.

## License

Apache 2.0
"""

readme_path = model_path / "README.md"
readme_path.write_text(readme_content)
print("✅ README created")

# Upload the model
print(f"\n📤 Uploading model to {repo_id}...")
print("This will take a while due to the large size...")

try:
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.pt", "*.pth", "*.cache", ".git*"]
    )
    print(f"\n✅ Model uploaded successfully!")
    print(f"View at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    sys.exit(1)