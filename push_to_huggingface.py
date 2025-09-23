#!/usr/bin/env python3
"""
Push Zen-Omni model to HuggingFace Hub.
Weights go to HuggingFace, code stays on GitHub.
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

def push_to_hf():
    api = HfApi()
    
    # Create repo on HuggingFace
    repo_id = "zenlm/zen-omni-30b"
    
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Created/verified repo: {repo_id}")
    except Exception as e:
        print(f"Repo exists or error: {e}")
    
    # Upload model files (after download completes)
    model_path = Path.home() / "work/zen/qwen3-omni-30b-complete"
    
    if model_path.exists():
        print(f"Uploading from {model_path}...")
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Zen-Omni 30B model weights"
        )
        print("✅ Upload complete!")
    else:
        print("⏳ Waiting for download to complete first...")

if __name__ == "__main__":
    push_to_hf()