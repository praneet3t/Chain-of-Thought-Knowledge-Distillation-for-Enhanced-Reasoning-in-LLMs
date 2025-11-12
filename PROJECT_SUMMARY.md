# Chain-of-Thought Distillation Project Summary

## Overview

This project implements a complete pipeline for Chain-of-Thought (CoT) knowledge distillation using Qwen models. It enables small student models to learn structured reasoning patterns from larger teacher models through efficient LoRA-based fine-tuning.

## Core Components

### 1. **helpers.py** (Utilities)
- `extract_final_answer()`: Extracts final answers from CoT reasoning using regex pattern matching
- Handles "Final Answer:" pattern with fallback to last non-empty line

### 2. **gen_cot.py** (CoT Generation)
- Loads teacher model (Qwen-14B/72B) from local directory
- Generates step-by-step reasoning for input questions
- Uses deterministic generation (temperature=0.2, do_sample=False)
- Outputs JSONL with question and cot fields

### 3. **train_cot_lora.py** (Student Training)
- Loads student model (Qwen-1.8B/7B) with 4-bit quantization
- Applies LoRA to attention and MLP layers (r=8, alpha=16)
- Formats data as "### Question:\n...\n\n### Answer:\n..."
- Trains with fp16, gradient accumulation, and paged optimizers
- Saves LoRA adapters to results directory

### 4. **eval_cot.py** (Evaluation)
- Loads base model + LoRA adapters
- Generates predictions on test set
- Extracts final answers for accuracy calculation
- Outputs predictions with gold answers (if available)

## Key Features

✅ **4-bit Quantization**: Enables training on consumer GPUs (8GB+ VRAM)  
✅ **LoRA Efficiency**: Only trains ~0.5% of parameters  
✅ **Deterministic Generation**: Consistent CoT reasoning from teacher  
✅ **Automatic Tokenizer Handling**: Falls back to eos_token for padding  
✅ **Progress Tracking**: tqdm progress bars for all stages  
✅ **Flexible Evaluation**: Works with or without gold answers  

## Technical Specifications

### LoRA Configuration
```python
r = 8                    # Rank of low-rank matrices
lora_alpha = 16          # Scaling factor
lora_dropout = 0.05      # Dropout probability
target_modules = [       # Modules to apply LoRA
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj"
]
```

### Training Configuration
```python
batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-4
num_epochs = 3
fp16 = True
optimizer = "paged_adamw_8bit"
max_length = 1024
```

### Generation Configuration
```python
temperature = 0.2
do_sample = False
max_new_tokens = 512
```

## Data Format

### Input Questions (raw_questions.jsonl)
```json
{"question": "What is 15% of 240?"}
```

### CoT Dataset (cot_dataset.jsonl)
```json
{"question": "What is 15% of 240?", "cot": "Step 1: ...\nFinal Answer: 36"}
```

### Test Questions (test_questions.jsonl)
```json
{"question": "What is 20% of 150?", "answer": "30"}
```

### Predictions (predictions.jsonl)
```json
{
  "question": "What is 20% of 150?",
  "generated_text": "Step 1: ...\nFinal Answer: 30",
  "prediction": "30",
  "gold_answer": "30"
}
```

## Memory Requirements

| Model Size | Quantization | Min VRAM | Recommended VRAM |
|------------|--------------|----------|------------------|
| 1.8B       | 4-bit        | 4GB      | 8GB              |
| 7B         | 4-bit        | 8GB      | 16GB             |
| 14B        | 4-bit        | 12GB     | 24GB             |

## Performance Expectations

- **Training Speed**: ~1-2 hours for 1000 examples (3 epochs, 7B model, single GPU)
- **Generation Speed**: ~1-2 min per 100 questions (teacher model)
- **Accuracy Improvement**: Typically 10-30% over baseline student model
- **Inference Speed**: Student model is 2-10x faster than teacher

## Command Reference

```bash
# Generate CoT
python scripts/gen_cot.py \
    --teacher_model models/teacher \
    --input_file data/raw_questions.jsonl \
    --output_file data/cot_dataset.jsonl

# Train Student
python scripts/train_cot_lora.py \
    --student_model models/student \
    --cot_dataset data/cot_dataset.jsonl \
    --output_dir results/student_cot_lora

# Evaluate
python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --test_file data/test_questions.jsonl \
    --output_file results/predictions.jsonl
```

## Dependencies

- torch >= 2.0.0
- transformers >= 4.35.0
- peft >= 0.7.0
- datasets >= 2.14.0
- bitsandbytes >= 0.41.0
- accelerate >= 0.24.0
- tqdm >= 4.65.0

## Project Structure

```
chain-of-thought-qwen/
├── data/                    # Input/output datasets
├── models/                  # Teacher and student models
│   ├── teacher/            # Large Qwen model
│   └── student/            # Small Qwen model
├── scripts/                # Python scripts
│   ├── helpers.py          # Utility functions
│   ├── gen_cot.py          # CoT generation
│   ├── train_cot_lora.py   # LoRA training
│   └── eval_cot.py         # Evaluation
├── results/                # Training outputs
├── README.md               # Full documentation
├── QUICKSTART.md           # Quick start guide
├── EXAMPLE_OUTPUT.md       # Example outputs
└── requirements.txt        # Dependencies
```

## Best Practices

1. **Start Small**: Test with 10-50 examples before scaling up
2. **Verify CoT Quality**: Manually inspect generated reasoning traces
3. **Monitor Training**: Check loss curves and sample outputs
4. **Tune Hyperparameters**: Adjust learning rate and epochs based on dataset size
5. **Use Validation Set**: Split data for better generalization assessment

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce batch_size to 1, increase gradient_accumulation |
| Poor Reasoning | Increase training epochs, verify CoT quality |
| Slow Generation | Reduce max_new_tokens, use smaller teacher model |
| No pad_token | Automatically handled by scripts |
| Low Accuracy | Generate more training examples, tune learning rate |

## License & Citation

This implementation is for educational and research purposes. Please cite relevant papers on chain-of-thought prompting and knowledge distillation when using this code.
