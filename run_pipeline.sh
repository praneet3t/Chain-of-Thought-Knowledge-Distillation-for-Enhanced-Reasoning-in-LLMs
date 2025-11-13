#!/bin/bash

# ============================================================================
# Chain-of-Thought Distillation Pipeline
# For HPC Cluster with 4x A100 GPUs (Offline Mode)
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "CHAIN-OF-THOUGHT DISTILLATION PIPELINE"
echo "============================================================================"
echo ""

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_MODEL="${PROJECT_ROOT}/models/teacher"
STUDENT_MODEL="${PROJECT_ROOT}/models/student"
RAW_QUESTIONS="${PROJECT_ROOT}/data/raw_questions.jsonl"
COT_DATASET="${PROJECT_ROOT}/data/cot_dataset.jsonl"
TEST_QUESTIONS="${PROJECT_ROOT}/data/test_questions.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/results/student_cot_lora"
PREDICTIONS="${PROJECT_ROOT}/results/predictions.jsonl"

# Check if models exist
if [ ! -d "$TEACHER_MODEL" ] || [ ! -f "$TEACHER_MODEL/config.json" ]; then
    echo "ERROR: Teacher model not found at $TEACHER_MODEL"
    echo "Please download models first using: python scripts/download_models.py"
    exit 1
fi

if [ ! -d "$STUDENT_MODEL" ] || [ ! -f "$STUDENT_MODEL/config.json" ]; then
    echo "ERROR: Student model not found at $STUDENT_MODEL"
    echo "Please download models first using: python scripts/download_models.py"
    exit 1
fi

echo "✓ Models found"
echo ""

# ============================================================================
# STAGE 1: Generate Chain-of-Thought Dataset
# ============================================================================
echo "============================================================================"
echo "STAGE 1: Generating Chain-of-Thought Reasoning"
echo "============================================================================"
echo "Using: 1 GPU (CUDA:0)"
echo "Input: $RAW_QUESTIONS"
echo "Output: $COT_DATASET"
echo ""

CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py \
    --teacher_model "$TEACHER_MODEL" \
    --input_file "$RAW_QUESTIONS" \
    --output_file "$COT_DATASET" \
    --max_new_tokens 512

echo ""
echo "✓ Stage 1 Complete: CoT dataset generated"
echo ""

# ============================================================================
# STAGE 2: Train Student Model with LoRA
# ============================================================================
echo "============================================================================"
echo "STAGE 2: Training Student Model with LoRA"
echo "============================================================================"
echo "Using: 4 GPUs (Multi-GPU Training)"
echo "Input: $COT_DATASET"
echo "Output: $OUTPUT_DIR"
echo ""

accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_cot_lora.py \
    --student_model "$STUDENT_MODEL" \
    --cot_dataset "$COT_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-4

echo ""
echo "✓ Stage 2 Complete: Student model trained"
echo ""

# ============================================================================
# STAGE 3: Evaluate Trained Model
# ============================================================================
echo "============================================================================"
echo "STAGE 3: Evaluating Trained Model"
echo "============================================================================"
echo "Using: 1 GPU (CUDA:0)"
echo "Input: $TEST_QUESTIONS"
echo "Output: $PREDICTIONS"
echo ""

CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py \
    --base_model "$STUDENT_MODEL" \
    --lora_model "$OUTPUT_DIR" \
    --test_file "$TEST_QUESTIONS" \
    --output_file "$PREDICTIONS" \
    --max_new_tokens 512

echo ""
echo "✓ Stage 3 Complete: Evaluation finished"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "============================================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================================"
echo ""
echo "Results saved to:"
echo "  - Trained model: $OUTPUT_DIR"
echo "  - Predictions: $PREDICTIONS"
echo ""
echo "To view predictions:"
echo "  head -n 5 $PREDICTIONS | python -m json.tool"
echo ""
