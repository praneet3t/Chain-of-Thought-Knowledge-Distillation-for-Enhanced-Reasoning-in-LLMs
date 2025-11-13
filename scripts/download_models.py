"""
Download models from HuggingFace Hub for offline use.
Run this script on the net node (with internet access).
"""

import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name: str, save_path: str):
    """Download model and tokenizer from HuggingFace Hub."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Save to: {save_path}")
    print(f"{'='*60}\n")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(save_path)
    print("✓ Tokenizer saved")
    
    # Download model
    print("Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.save_pretrained(save_path)
    print("✓ Model saved")
    
    print(f"\n✓ Successfully downloaded {model_name}\n")


if __name__ == "__main__":
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Download teacher model (Qwen-14B-Chat)
    teacher_path = project_root / "models" / "teacher"
    download_model("Qwen/Qwen-14B-Chat", str(teacher_path))
    
    # Download student model (Qwen-7B)
    student_path = project_root / "models" / "student"
    download_model("Qwen/Qwen-7B", str(student_path))
    
    print("\n" + "="*60)
    print("ALL MODELS DOWNLOADED SUCCESSFULLY")
    print("="*60)
    print("\nYou can now run the pipeline on the offline GPU node.")
    print("\nNext steps:")
    print("1. Switch to GPU node")
    print("2. Run: python scripts/gen_cot.py ...")
    print("3. Run: accelerate launch scripts/train_cot_lora.py ...")
    print("4. Run: python scripts/eval_cot.py ...")
