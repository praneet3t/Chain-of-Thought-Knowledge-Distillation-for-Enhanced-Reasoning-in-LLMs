#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Activate conda env: 'conda activate cot_distill' (make sure shared env is available)"
echo "Starting pipeline..."

# Stage 1: Generate CoT
echo "Stage 1: Generate CoT"
CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py --teacher_dir models/teacher --input data/raw_questions.jsonl --output data/cot_dataset.jsonl --max_new_tokens 512

# Stage 2: Train LoRA (distributed)
echo "Stage 2: Train LoRA"
accelerate launch --config_file configs/accelerate_config.yaml scripts/train_cot_lora.py --student_dir models/student --data data/cot_dataset.jsonl --out_dir results/student_cot_lora --per_device_train_batch_size 2 --epochs 3

# Stage 3: Evaluate
echo "Stage 3: Evaluate"
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py --base_model models/student --lora_model results/student_cot_lora --input data/test_questions.jsonl --output results/predictions.jsonl --max_new_tokens 256

echo "Pipeline complete."
