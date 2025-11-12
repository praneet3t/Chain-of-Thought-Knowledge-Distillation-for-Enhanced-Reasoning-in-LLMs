# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Complete Pipeline (3 Commands)

### 1. Generate CoT Dataset
```bash
python scripts/gen_cot.py \
    --teacher_model models/teacher \
    --input_file data/raw_questions.jsonl \
    --output_file data/cot_dataset.jsonl
```

### 2. Train Student Model
```bash
python scripts/train_cot_lora.py \
    --student_model models/student \
    --cot_dataset data/cot_dataset.jsonl \
    --output_dir results/student_cot_lora
```

### 3. Evaluate Model
```bash
python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --test_file data/test_questions.jsonl \
    --output_file results/predictions.jsonl
```

## Before You Start

1. **Download Models**: Place Qwen teacher model in `models/teacher/` and student model in `models/student/`
2. **Prepare Data**: Add your questions to `data/raw_questions.jsonl`
3. **Check GPU**: Ensure you have at least 8GB VRAM for 7B models

## Expected Timeline

- **CoT Generation**: ~1-2 minutes per 100 questions (depends on teacher model size)
- **Training**: ~30-60 minutes for 3 epochs on 1000 examples (with 4-bit quantization)
- **Evaluation**: ~30 seconds per 100 questions

## Minimal Example

Test with the provided sample data (5 questions):

```bash
# Step 1: Generate (requires teacher model)
python scripts/gen_cot.py --teacher_model models/teacher --input_file data/raw_questions.jsonl --output_file data/cot_dataset.jsonl

# Step 2: Train (requires student model)
python scripts/train_cot_lora.py --student_model models/student --cot_dataset data/cot_dataset.jsonl --output_dir results/student_cot_lora --num_epochs 1

# Step 3: Evaluate
python scripts/eval_cot.py --base_model models/student --lora_model results/student_cot_lora --test_file data/test_questions.jsonl --output_file results/predictions.jsonl
```

## Troubleshooting

**Out of Memory?**
```bash
# Use smaller batch size
python scripts/train_cot_lora.py ... --batch_size 1
```

**No pad_token warning?**
- Ignore it - scripts handle this automatically

**Want faster training?**
```bash
# Reduce epochs for testing
python scripts/train_cot_lora.py ... --num_epochs 1
```
