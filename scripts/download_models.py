# scripts/download_models.py
import argparse
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

def download_repo(repo_id: str, dest: Path, allow_patterns=None, repo_type="model", force=False):
    dest = dest.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading repo '{repo_id}' -> {dest}")
    cache_path = snapshot_download(
        repo_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        resume_download=True,
        force_download=force,
    )
    print(f"Snapshot in HF cache: {cache_path}")
    print("Copying to destination (merge)...")
    shutil.copytree(cache_path, dest, dirs_exist_ok=True)
    print("Copied.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="Qwen/Qwen-14B-Chat")
    parser.add_argument("--student", default="Qwen/Qwen-7B")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)

    # Patterns - only typical files (saves time)
    patterns = [
        "*.json", "*.txt", "tokenizer*", "special_tokens_map.json",
        "*.bin", "*.safetensors", "pytorch_model*.bin", "pytorch_model*.safetensors",
        "config.json", "generation_config.json", "*.model"
    ]

    teacher_dest = out / "teacher"
    student_dest = out / "student"

    print("Downloading teacher model...")
    download_repo(args.teacher, teacher_dest, allow_patterns=patterns, force=args.force)

    print("Downloading student model...")
    download_repo(args.student, student_dest, allow_patterns=patterns, force=args.force)

    print("\n✅ All done. Verify the folders under", out)

if __name__ == "__main__":
    main()
