# scripts/download_models_safe.py
"""
Download model repository files from Hugging Face to a local folder WITHOUT loading into memory.
Run on the net node (internet access).
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model_files(repo_id: str, dest_dir: str, allow_patterns=None):
    """
    Uses snapshot_download to fetch model repo files. Copies result into dest_dir.
    - repo_id: 'Qwen/Qwen-14B-Chat'
    - dest_dir: local path to copy files into
    - allow_patterns: list of glob patterns (optional) to restrict what is downloaded
    """
    dest = Path(dest_dir).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading repo `{repo_id}` into temporary cache (may take long)...")
    # snapshot_download will place the repo under HF cache directory; returns path to that folder.
    cache_path = snapshot_download(
        repo_id,
        repo_type="model",
        allow_patterns=allow_patterns,  # None downloads everything
        resume_download=True,
        force_download=False,
    )
    print(f"Snapshot downloaded to cache: {cache_path}")
    # copy to dest (merge if dest exists)
    print(f"Copying files to final destination: {dest}")
    shutil.copytree(cache_path, dest, dirs_exist_ok=True)
    print(f"Done. Files available at: {dest}\n")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    models_root = project_root / "models"
    teacher_dest = models_root / "teacher"
    student_dest = models_root / "student"

    # OPTIONAL: restrict to typical model files - helps skip large unrelated files
    patterns = [
        "*.json", "config.json", "tokenizer*", "*.txt", "*.py",
        "*.bin", "*.safetensors", "pytorch_model*.bin", "pytorch_model*.safetensors",
        "generation_config.json", "special_tokens_map.json", "vocab.json"
    ]

    print("="*60)
    print("Downloading teacher model repo (Qwen-14B-Chat)...")
    download_model_files("Qwen/Qwen-14B-Chat", str(teacher_dest), allow_patterns=patterns)

    print("Downloading student model repo (Qwen-7B)...")
    download_model_files("Qwen/Qwen-7B", str(student_dest), allow_patterns=patterns)

    print("\nALL MODELS DOWNLOADED (files only).")
    print("Now copy/verify these folders are visible to your offline GPU node.")
    print("="*60)
