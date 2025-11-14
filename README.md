# Chain-of-Thought Knowledge Distillation

## Overview

This project implements chain-of-thought (CoT) knowledge distillation for training smaller language models to perform step-by-step reasoning. A large teacher model generates detailed reasoning traces for a dataset of questions, which are then used to fine-tune a smaller student model using parameter-efficient LoRA adapters.

The implementation is designed for HPC clusters with offline GPU nodes and supports distributed multi-GPU training via Hugging Face Accelerate.

## Architecture

### Pipeline Stages

1. **CoT Generation**: Teacher model generates step-by-step reasoning traces for input questions
2. **LoRA Training**: Student model is fine-tuned on generated CoT data using low-rank adaptation
3. **Evaluation**: Trained model is evaluated on held-out test set with accuracy metrics

### Models

- **Teacher**: Qwen-14B-Chat (14B parameters, ~28GB disk space)
- **Student**: Qwen-7B (7B parameters, ~14GB disk space)

### Training Method

- **Quantization**: 4-bit NF4 quantization via bitsandbytes
- **PEFT**: LoRA (Low-Rank Adaptation) with rank r=8, alpha=16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj
- **Optimizer**: paged_adamw_8bit for memory efficiency

## System Requirements

### Hardware

- Minimum: 1x A100 40GB GPU
- Recommended: 4x A100 80GB GPUs for distributed training
- Storage: 50GB free space for models and data

### Software

- Python 3.10+
- CUDA 11.8+
- Conda or virtualenv
- Network access on download node (for model acquisition)
- Offline capability on GPU compute node

## Installation

### On Network-Connected Node

```bash
# Create environment
conda create -n cot_distill python=3.10 -y
conda activate cot_distill

# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install transformers==4.36.0 \
    peft==0.7.1 \
    datasets==2.16.0 \
    bitsandbytes==0.41.3 \
    accelerate==0.25.0 \
    tqdm==4.66.1 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    deepspeed==0.12.6 \
    huggingface-hub
```

### Download Models

```bash
cd /path/to/chain-of-thought-qwen
python scripts/download_models.py --teacher Qwen/Qwen-14B-Chat --student Qwen/Qwen-7B --out_dir models
```

This downloads models to `models/teacher/` and `models/student/` directories. All subsequent operations use `local_files_only=True` to prevent network access.

## Project Structure

```
chain-of-thought-qwen/
├── configs/
│   └── accelerate_config.yaml       # Multi-GPU training configuration
├── data/
│   ├── raw_questions.jsonl          # Input questions (377 samples)
│   ├── cot_dataset.jsonl            # Generated CoT traces
│   └── test_questions.jsonl         # Test set with gold answers (95 samples)
├── models/
│   ├── teacher/                     # Qwen-14B-Chat
│   └── student/                     # Qwen-7B
├── scripts/
│   ├── helpers.py                   # Utility functions
│   ├── gen_cot.py                   # CoT generation script
│   ├── train_cot_lora.py            # LoRA training script
│   ├── eval_cot.py                  # Evaluation script
│   ├── download_models.py           # Model download utility
│   └── generate_dataset.py          # Dataset generation utility
├── results/
│   ├── student_cot_lora/            # Trained LoRA adapters
│   └── predictions.jsonl            # Evaluation outputs
└── run_pipeline.sh                  # Automated pipeline execution
```

## Usage

### Automated Execution

```bash
conda activate cot_distill
cd /path/to/chain-of-thought-qwen
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Manual Execution

#### Stage 1: Generate Chain-of-Thought Dataset

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py \
    --teacher_dir models/teacher \
    --input data/raw_questions.jsonl \
    --output data/cot_dataset.jsonl \
    --max_new_tokens 512
```

**Parameters:**
- `--teacher_dir`: Path to teacher model directory
- `--input`: Input JSONL file with `question` field
- `--output`: Output JSONL file with `question` and `cot` fields
- `--max_new_tokens`: Maximum tokens to generate per question
- `--start_index`: Resume from specific index (for interrupted runs)
- `--num_samples`: Limit number of samples to process

**Output Format:**
```json
{"question": "What is 15% of 240?", "cot": "Step 1: Convert 15% to decimal...\nFinal Answer: 36"}
```

**Time**: 10-15 minutes for 377 questions on A100

#### Stage 2: Train Student Model

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_cot_lora.py \
    --student_dir models/student \
    --data data/cot_dataset.jsonl \
    --out_dir results/student_cot_lora \
    --per_device_train_batch_size 2 \
    --epochs 3 \
    --lr 2e-4
```

**Parameters:**
- `--student_dir`: Path to student model directory
- `--data`: CoT dataset JSONL file
- `--out_dir`: Output directory for LoRA adapters
- `--per_device_train_batch_size`: Batch size per GPU
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha scaling (default: 16)
- `--max_length`: Maximum sequence length (default: 1024)
- `--resume_from_checkpoint`: Resume from checkpoint directory

**Training Format:**
```
### Question:
What is 15% of 240?

### Answer:
Step 1: Convert 15% to decimal...
Final Answer: 36
```

**Time**: 30-45 minutes for 377 samples, 3 epochs on 4x A100

#### Stage 3: Evaluate Model

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --input data/test_questions.jsonl \
    --output results/predictions.jsonl \
    --max_new_tokens 256
```

**Parameters:**
- `--base_model`: Path to base student model
- `--lora_model`: Path to LoRA adapter directory (optional)
- `--input`: Test JSONL file with `question` and optional `answer` fields
- `--output`: Output JSONL file with predictions
- `--max_new_tokens`: Maximum tokens to generate

**Output Format:**
```json
{"question": "What is 20% of 150?", "generated": "Step 1: ...\nFinal Answer: 30", "pred": "30", "gold": "30"}
```

**Time**: 2-3 minutes for 95 questions on A100

## Dataset

### Training Set

377 questions across 6 categories:
- Arithmetic: 100 (addition, subtraction, multiplication, division, mixed operations)
- Percentages: 80 (percentage calculations)
- Algebra: 80 (linear equations)
- Word Problems: 120 (speed/distance, area, price, age problems)
- Fractions: 60 (fraction operations and arithmetic)
- Ratios: 60 (ratio and proportion problems)

### Test Set

95 questions with gold answers for accuracy evaluation, sampled from the same distribution.

### Dataset Generation

```bash
python scripts/generate_dataset.py
```

Generates `data/raw_questions.jsonl` (training) and `data/test_questions.jsonl` (test).

## Configuration

### Multi-GPU Training (configs/accelerate_config.yaml)

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
gpu_ids: all
mixed_precision: fp16
```

Modify `num_processes` to match available GPU count.

### LoRA Configuration

Default parameters in `train_cot_lora.py`:
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: All attention and MLP projection layers

### Training Hyperparameters

- Learning rate: 2e-4
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Effective batch size: 2 × 4 GPUs × 4 accumulation = 32
- Epochs: 3
- Optimizer: paged_adamw_8bit
- Precision: FP16
- Max sequence length: 1024 tokens

## Technical Details

### Offline Mode

All model loading operations use `local_files_only=True` to prevent network access:

```python
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
```

This ensures the pipeline runs on air-gapped GPU nodes after models are downloaded on a network-connected node.

### Memory Optimization

1. **4-bit Quantization**: Models loaded with `load_in_4bit=True` using NF4 quantization
2. **LoRA**: Only trains ~0.5% of parameters (rank-8 low-rank matrices)
3. **Gradient Accumulation**: Reduces per-step memory by accumulating over 4 steps
4. **Paged Optimizer**: Uses paged_adamw_8bit to offload optimizer states

### Answer Extraction

The `extract_final_answer()` function in `helpers.py` uses regex to locate answer markers:
1. Searches for "Final Answer:", "Answer:", or similar patterns (case-insensitive)
2. Returns first line after marker
3. Falls back to last non-empty line if no marker found

### Prompt Format

**Generation (Teacher):**
```
You are a careful reasoning assistant. Solve the problem step-by-step and end with a line that starts with 'Final Answer:'.

Q: {question}

A:
```

**Training/Evaluation (Student):**
```
### Question:
{question}

### Answer:
{cot}
```

## Performance

### Expected Results

- Training loss: Decreases from ~2.5 to ~0.5 over 3 epochs
- Test accuracy: 70-85% exact match on math problems
- LoRA adapter size: ~50MB (vs 14GB full model)
- Inference speed: ~2-3 tokens/second per question on A100

### Benchmarks

| Stage | Time | GPU Usage | Memory |
|-------|------|-----------|--------|
| CoT Generation | 10-15 min | 1x A100 | ~20GB |
| Training | 30-45 min | 4x A100 | ~30GB per GPU |
| Evaluation | 2-3 min | 1x A100 | ~20GB |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
--per_device_train_batch_size 1
```

Or reduce sequence length in `train_cot_lora.py`:
```python
--max_length 512
```

### Missing pad_token

Automatically handled by setting `tokenizer.pad_token = tokenizer.eos_token` in all scripts.

### Connection Errors on GPU Node

Ensure models are downloaded on network-connected node first. All scripts use `local_files_only=True`.

### Slow Generation

Normal for autoregressive generation. Teacher model processes questions sequentially. Consider reducing `--max_new_tokens` to 256-384 for faster generation.

### Training Not Converging

- Increase epochs: `--epochs 5`
- Adjust learning rate: `--lr 1e-4` or `--lr 3e-4`
- Verify CoT quality in `data/cot_dataset.jsonl`
- Increase dataset size

### Resume Interrupted Training

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_cot_lora.py \
    --resume_from_checkpoint results/student_cot_lora/checkpoint-500 \
    [other args...]
```

## Advanced Usage

### Custom Models

Replace teacher/student models by downloading different models:

```bash
python scripts/download_models.py \
    --teacher meta-llama/Llama-2-13b-chat-hf \
    --student meta-llama/Llama-2-7b-hf \
    --out_dir models
```

### Custom Dataset

Create JSONL file with `question` field:

```json
{"question": "Your question here"}
```

Then run pipeline with `--input your_questions.jsonl`.

### Hyperparameter Tuning

Modify LoRA rank and alpha:

```bash
python scripts/train_cot_lora.py \
    --r 16 \
    --lora_alpha 32 \
    [other args...]
```

Higher rank increases capacity but also memory usage and training time.

### Distributed Training Configuration

Edit `configs/accelerate_config.yaml` for different GPU counts:

```yaml
num_processes: 2  # For 2 GPUs
```

Or generate new config:

```bash
accelerate config
```

## File Formats

### Input Questions (raw_questions.jsonl)

```json
{"question": "What is 15% of 240?"}
```

### CoT Dataset (cot_dataset.jsonl)

```json
{"question": "What is 15% of 240?", "cot": "Step 1: Convert percentage...\nFinal Answer: 36"}
```

### Test Questions (test_questions.jsonl)

```json
{"question": "What is 20% of 150?", "answer": "30"}
```

### Predictions (predictions.jsonl)

```json
{"question": "What is 20% of 150?", "generated": "Step 1: ...\nFinal Answer: 30", "pred": "30", "gold": "30"}
```

## Dependencies

Core packages:
- torch >= 2.0.0
- transformers >= 4.36.0
- peft >= 0.7.0
- datasets >= 2.16.0
- bitsandbytes >= 0.41.0
- accelerate >= 0.25.0

See `requirements.txt` for complete list.


