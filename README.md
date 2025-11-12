# Chain-of-Thought Knowledge Distillation for Enhanced Reasoning in LLMs

This project implements Chain-of-Thought (CoT) distillation to enhance small student models by fine-tuning them on step-by-step reasoning traces generated from larger teacher models. The student learns structured reasoning patterns through efficient LoRA-based training on synthetic CoT datasets.

## Project Structure

```
chain-of-thought-qwen/
├── data/
│   ├── raw_questions.jsonl       # Input questions
│   ├── cot_dataset.jsonl         # Generated CoT reasoning
│   └── test_questions.jsonl      # Test set
├── models/
│   ├── teacher/                  # Teacher model directory
│   └── student/                  # Student model directory
├── scripts/
│   ├── helpers.py                # Utility functions
│   ├── gen_cot.py                # Generate CoT from teacher
│   ├── train_cot_lora.py         # Train student with LoRA
│   └── eval_cot.py               # Evaluate trained model
├── results/                      # Training outputs
└── README.md
```

## Setup Instructions

### 1. Environment Setup

```bash
pip install torch transformers peft datasets bitsandbytes accelerate tqdm
```

### 2. Model Placement

Download Qwen models and place them in the appropriate directories:

- **Teacher Model**: Place in `models/teacher/` (e.g., Qwen-14B or Qwen-72B)
- **Student Model**: Place in `models/student/` (e.g., Qwen-1.8B or Qwen-7B)

Example structure:
```
models/
├── teacher/
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── pytorch_model.bin (or model.safetensors)
│   └── ...
└── student/
    ├── config.json
    ├── tokenizer_config.json
    ├── pytorch_model.bin (or model.safetensors)
    └── ...
```

### 3. Prepare Input Data

Create `data/raw_questions.jsonl` with your questions:

```jsonl
{"question": "What is 15% of 240?"}
{"question": "If a train travels 120 km in 2 hours, what is its average speed?"}
{"question": "Solve for x: 2x + 5 = 13"}
```

For evaluation with gold answers, create `data/test_questions.jsonl`:

```jsonl
{"question": "What is 15% of 240?", "answer": "36"}
{"question": "If a train travels 120 km in 2 hours, what is its average speed?", "answer": "60 km/h"}
```

## Usage Pipeline

### Stage 1: Generate Chain-of-Thought Reasoning

Use the teacher model to generate step-by-step reasoning for each question:

```bash
python scripts/gen_cot.py \
    --teacher_model models/teacher \
    --input_file data/raw_questions.jsonl \
    --output_file data/cot_dataset.jsonl \
    --max_new_tokens 512
```

**Output**: `data/cot_dataset.jsonl` containing questions with CoT reasoning traces.

### Stage 2: Train Student Model with LoRA

Fine-tune the student model on the generated CoT dataset:

```bash
python scripts/train_cot_lora.py \
    --student_model models/student \
    --cot_dataset data/cot_dataset.jsonl \
    --output_dir results/student_cot_lora \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

**Output**: Fine-tuned LoRA weights in `results/student_cot_lora/`

### Stage 3: Evaluate Trained Model

Test the fine-tuned model on held-out questions:

```bash
python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --test_file data/test_questions.jsonl \
    --output_file results/predictions.jsonl \
    --max_new_tokens 512
```

**Output**: Predictions and accuracy metrics in `results/predictions.jsonl`

## Hyperparameter Recommendations

### LoRA Configuration
- **r**: 8 (rank of LoRA matrices)
- **lora_alpha**: 16 (scaling factor)
- **lora_dropout**: 0.05
- **target_modules**: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj

### Training Parameters
- **Learning rate**: 2e-4 (adjust to 1e-4 for larger models)
- **Batch size**: 4 (reduce to 1-2 for memory constraints)
- **Gradient accumulation**: 4 steps
- **Epochs**: 3 (increase to 5 for smaller datasets)
- **Max sequence length**: 1024 tokens

### Generation Parameters
- **Temperature**: 0.2 (deterministic reasoning)
- **do_sample**: False
- **max_new_tokens**: 512 (adjust based on reasoning complexity)

## Memory Optimization

### Low-Resource Scenarios

**For 4-bit quantization (default)**:
- Minimum GPU: 8GB VRAM for 7B student model
- Recommended: 16GB+ VRAM

**Additional optimizations**:

1. **Reduce batch size**:
   ```bash
   --batch_size 1 --gradient_accumulation_steps 8
   ```

2. **Reduce sequence length** in `train_cot_lora.py`:
   ```python
   max_length=512  # instead of 1024
   ```

3. **Use CPU offloading**:
   ```python
   device_map="auto"  # Already enabled
   ```

4. **Enable gradient checkpointing** (add to training args):
   ```python
   gradient_checkpointing=True
   ```

## Troubleshooting

### Issue: "Tokenizer has no pad_token"

**Solution**: The scripts automatically set `pad_token = eos_token`. If issues persist, manually add to tokenizer config:
```python
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
```

### Issue: CUDA Out of Memory

**Solutions**:
1. Reduce batch size: `--batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 8`
3. Reduce max_length in tokenization (edit `train_cot_lora.py`)
4. Use smaller student model (e.g., Qwen-1.8B instead of Qwen-7B)

### Issue: Slow generation during CoT creation

**Solutions**:
1. Use smaller teacher model if accuracy permits
2. Reduce `--max_new_tokens` to 256-384
3. Process in batches (modify `gen_cot.py` to batch inputs)

### Issue: Model not learning reasoning patterns

**Solutions**:
1. Increase training epochs: `--num_epochs 5`
2. Verify CoT quality in `data/cot_dataset.jsonl`
3. Increase dataset size (generate more examples)
4. Adjust learning rate: try `--learning_rate 1e-4` or `3e-4`

### Issue: Poor final answer extraction

**Solution**: The `extract_final_answer()` function looks for "Final Answer:" pattern. Ensure teacher model consistently uses this format. Modify prompt if needed.

## Shared Filesystem Compatibility

All paths use relative references to support shared filesystems between nodes:
- Models loaded from local directories
- No hardcoded absolute paths
- Results saved to local `results/` directory

For distributed training, ensure:
1. All nodes have access to model directories
2. Output directory is writable from all nodes
3. Use `CUDA_VISIBLE_DEVICES` to control GPU allocation

## Expected Results

After successful training:
- Student model generates step-by-step reasoning
- Improved accuracy on reasoning tasks (10-30% improvement typical)
- Structured output format matching training data
- Faster inference than teacher model with comparable reasoning quality

## Citation

If you use this implementation, please cite the relevant papers on chain-of-thought prompting and knowledge distillation.
