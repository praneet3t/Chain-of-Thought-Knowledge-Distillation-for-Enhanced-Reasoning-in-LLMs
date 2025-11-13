# Complete Execution Guide for HPC Cluster

## System Setup
- **Hardware**: 4x A100 80GB GPUs
- **Network**: Offline GPU node + Online net node
- **Storage**: Shared filesystem

---

## PART 1: Setup on Net Node (With Internet)

### Step 1: Create Environment
```bash
conda create -n cot_distill python=3.10 -y
conda activate cot_distill
```

### Step 2: Install Dependencies
```bash
# Install PyTorch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
pip install transformers==4.36.0 peft==0.7.1 datasets==2.16.0 \
    bitsandbytes==0.41.3 accelerate==0.25.0 tqdm==4.66.1 \
    sentencepiece==0.1.99 protobuf==3.20.3 deepspeed==0.12.6
```

### Step 3: Navigate to Project
```bash
cd /path/to/shared/filesystem/chain-of-thought-qwen
```

### Step 4: Download Models
```bash
python scripts/download_models.py
```

This downloads:
- Qwen/Qwen-14B-Chat → models/teacher/
- Qwen/Qwen-7B → models/student/

**Time**: 20-30 minutes

### Step 5: Verify Downloads
```bash
ls -lh models/teacher/
ls -lh models/student/
```

---

## PART 2: Run on GPU Node (Offline)

### Step 1: Activate Environment
```bash
conda activate cot_distill
cd /path/to/shared/filesystem/chain-of-thought-qwen
```

### Step 2: Generate Dataset (Optional)
```bash
# Already generated, but you can regenerate:
python scripts/generate_dataset.py
```

### Step 3: Run Complete Pipeline
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**OR run each stage manually:**

#### Stage 1: Generate CoT (10-15 min)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py \
    --teacher_model models/teacher \
    --input_file data/raw_questions.jsonl \
    --output_file data/cot_dataset.jsonl \
    --max_new_tokens 512
```

#### Stage 2: Train Student (30-45 min)
```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_cot_lora.py \
    --student_model models/student \
    --cot_dataset data/cot_dataset.jsonl \
    --output_dir results/student_cot_lora \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-4
```

#### Stage 3: Evaluate (2-3 min)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --test_file data/test_questions.jsonl \
    --output_file results/predictions.jsonl \
    --max_new_tokens 512
```

---

## Monitoring

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View Training Progress
```bash
tail -f results/student_cot_lora/logs/events.out.tfevents.*
```

### Check Results
```bash
# View predictions
head -n 3 results/predictions.jsonl | python -m json.tool

# Count correct predictions
grep -o '"gold_answer"' results/predictions.jsonl | wc -l
```

---

## Expected Timeline

| Stage | Time | GPUs Used |
|-------|------|-----------|
| Download Models | 20-30 min | N/A (net node) |
| Generate CoT | 10-15 min | 1x A100 |
| Train Student | 30-45 min | 4x A100 |
| Evaluate | 2-3 min | 1x A100 |
| **Total** | **~1-1.5 hours** | |

---

## File Structure After Completion

```
chain-of-thought-qwen/
├── data/
│   ├── raw_questions.jsonl      (377 questions)
│   ├── cot_dataset.jsonl        (377 with reasoning)
│   └── test_questions.jsonl     (95 questions)
├── models/
│   ├── teacher/                 (~28GB)
│   └── student/                 (~14GB)
├── results/
│   ├── student_cot_lora/        (LoRA weights ~50MB)
│   └── predictions.jsonl        (95 predictions)
```

---

## Troubleshooting

**Error: "No module named 'transformers'"**
- Solution: Activate conda environment

**Error: "Connection timeout"**
- Solution: Ensure models downloaded on net node first

**Error: "CUDA out of memory"**
- Solution: Reduce batch_size to 4 or 2

**Error: "Model not found"**
- Solution: Check models/teacher/ and models/student/ exist

---

## Quick Commands Reference

```bash
# On net node (with internet)
conda activate cot_distill
cd /path/to/project
python scripts/download_models.py

# On GPU node (offline)
conda activate cot_distill
cd /path/to/project
./run_pipeline.sh

# Or run stages individually
CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py --teacher_model models/teacher --input_file data/raw_questions.jsonl --output_file data/cot_dataset.jsonl
accelerate launch --config_file configs/accelerate_config.yaml scripts/train_cot_lora.py --student_model models/student --cot_dataset data/cot_dataset.jsonl --output_dir results/student_cot_lora
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py --base_model models/student --lora_model results/student_cot_lora --test_file data/test_questions.jsonl --output_file results/predictions.jsonl
```
