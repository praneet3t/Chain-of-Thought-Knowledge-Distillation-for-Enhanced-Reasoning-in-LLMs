# HPC Cluster Setup Guide (Offline Mode)

## System Configuration
- **GPUs**: 4x A100 80GB
- **Network**: Offline GPU node + Online "net node"
- **Storage**: Shared filesystem accessible from both nodes
- **Environment**: Conda

---

## STEP 1: Setup on Net Node (With Internet)

### 1.1 Create Conda Environment
```bash
# On net node
conda create -n cot_distill python=3.10 -y
conda activate cot_distill

# Install PyTorch with CUDA 11.8
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install transformers==4.36.0 peft==0.7.1 datasets==2.16.0 \
    bitsandbytes==0.41.3 accelerate==0.25.0 tqdm==4.66.1 \
    sentencepiece==0.1.99 protobuf==3.20.3 deepspeed==0.12.6
```

### 1.2 Download Models
```bash
# Navigate to project
cd /path/to/shared/filesystem/chain-of-thought-qwen

# Download models using Python
python scripts/download_models.py
```

This will download:
- **Teacher**: Qwen/Qwen-14B-Chat
- **Student**: Qwen/Qwen-7B

Models will be saved to `models/teacher/` and `models/student/`

### 1.3 Verify Downloads
```bash
# Check models are downloaded
ls -lh models/teacher/
ls -lh models/student/

# Should see config.json, tokenizer files, and model weights
```

---

## STEP 2: Run on GPU Node (Offline)

### 2.1 Activate Environment
```bash
# On GPU node
conda activate cot_distill
cd /path/to/shared/filesystem/chain-of-thought-qwen
```

### 2.2 Generate CoT Dataset (Single GPU)
```bash
# Use 1 GPU for generation
CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py \
    --teacher_model models/teacher \
    --input_file data/raw_questions.jsonl \
    --output_file data/cot_dataset.jsonl \
    --max_new_tokens 512
```

**Time**: ~10-15 minutes for 500 questions

### 2.3 Train Student Model (Multi-GPU)
```bash
# Use all 4 GPUs with DeepSpeed
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_cot_lora.py \
    --student_model models/student \
    --cot_dataset data/cot_dataset.jsonl \
    --output_dir results/student_cot_lora \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-4
```

**Time**: ~30-45 minutes for 500 examples (3 epochs)

### 2.4 Evaluate Model (Single GPU)
```bash
# Use 1 GPU for evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --test_file data/test_questions.jsonl \
    --output_file results/predictions.jsonl \
    --max_new_tokens 512
```

**Time**: ~2-3 minutes for 100 test questions

---

## STEP 3: Monitor and Verify

### Check Training Progress
```bash
# View training logs
tail -f results/student_cot_lora/trainer_log.txt

# Check GPU utilization
watch -n 1 nvidia-smi
```

### Verify Results
```bash
# Check accuracy
cat results/predictions.jsonl | grep "gold_answer"

# View sample predictions
head -n 3 results/predictions.jsonl | python -m json.tool
```

---

## Complete Execution Order

```bash
# ============================================
# ON NET NODE (with internet)
# ============================================
conda create -n cot_distill python=3.10 -y
conda activate cot_distill
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers peft datasets bitsandbytes accelerate tqdm sentencepiece protobuf deepspeed
cd /path/to/shared/filesystem/chain-of-thought-qwen
python scripts/download_models.py

# ============================================
# ON GPU NODE (offline)
# ============================================
conda activate cot_distill
cd /path/to/shared/filesystem/chain-of-thought-qwen

# Step 1: Generate CoT (10-15 min)
CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py \
    --teacher_model models/teacher \
    --input_file data/raw_questions.jsonl \
    --output_file data/cot_dataset.jsonl

# Step 2: Train Student (30-45 min)
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_cot_lora.py \
    --student_model models/student \
    --cot_dataset data/cot_dataset.jsonl \
    --output_dir results/student_cot_lora \
    --num_epochs 3 \
    --batch_size 8

# Step 3: Evaluate (2-3 min)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --test_file data/test_questions.jsonl \
    --output_file results/predictions.jsonl
```

---

## Troubleshooting

### Issue: "No module named 'transformers'"
**Solution**: Ensure conda environment is activated on GPU node

### Issue: "Connection timeout" on GPU node
**Solution**: All models must be pre-downloaded on net node first

### Issue: Out of memory with 4 GPUs
**Solution**: Reduce batch_size to 4 or 2 in training command

### Issue: Slow generation
**Solution**: Normal - teacher model generation is sequential. Consider using vLLM for faster inference.

---

## Expected Results

- **Dataset Size**: 500 training examples, 100 test examples
- **Training Time**: ~30-45 minutes (3 epochs, 4x A100)
- **Final Accuracy**: 70-85% on math reasoning tasks
- **Model Size**: LoRA adapters ~50MB (vs 14GB full model)
