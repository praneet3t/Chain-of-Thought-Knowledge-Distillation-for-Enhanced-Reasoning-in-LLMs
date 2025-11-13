# Complete Order of Execution

## Overview
This guide shows the EXACT order to run everything for your HPC cluster setup.

---

## PHASE 1: Net Node (With Internet Access) - ONE TIME ONLY

### Step 1: Create Conda Environment
```bash
conda create -n cot_distill python=3.10 -y
```

### Step 2: Activate Environment
```bash
conda activate cot_distill
```

### Step 3: Install PyTorch
```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Step 4: Install Python Packages
```bash
pip install transformers==4.36.0 peft==0.7.1 datasets==2.16.0 \
    bitsandbytes==0.41.3 accelerate==0.25.0 tqdm==4.66.1 \
    sentencepiece==0.1.99 protobuf==3.20.3 deepspeed==0.12.6
```

### Step 5: Navigate to Project
```bash
cd /path/to/shared/filesystem/chain-of-thought-qwen
```
Replace `/path/to/shared/filesystem/` with your actual path.

### Step 6: Download Models
```bash
python scripts/download_models.py
```

**What this does:**
- Downloads Qwen-14B-Chat to `models/teacher/`
- Downloads Qwen-7B to `models/student/`
- Takes 20-30 minutes
- Requires ~42GB disk space

### Step 7: Verify Downloads
```bash
ls -lh models/teacher/
ls -lh models/student/
```

You should see files like:
- config.json
- tokenizer_config.json
- model.safetensors (or pytorch_model.bin)

**✅ PHASE 1 COMPLETE - You can now switch to GPU node**

---

## PHASE 2: GPU Node (Offline) - ACTUAL TRAINING

### Step 1: Activate Environment
```bash
conda activate cot_distill
```

### Step 2: Navigate to Project
```bash
cd /path/to/shared/filesystem/chain-of-thought-qwen
```

### Step 3: Verify Models Exist
```bash
ls models/teacher/config.json
ls models/student/config.json
```

Both should exist. If not, go back to Phase 1.

### Step 4: Check Dataset
```bash
wc -l data/raw_questions.jsonl
wc -l data/test_questions.jsonl
```

Should show:
- 377 raw_questions.jsonl
- 95 test_questions.jsonl

### Step 5: Run Complete Pipeline
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**OR run each stage manually:**

---

## MANUAL EXECUTION (Alternative to Step 5)

### Stage 1: Generate Chain-of-Thought Dataset

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py \
    --teacher_model models/teacher \
    --input_file data/raw_questions.jsonl \
    --output_file data/cot_dataset.jsonl \
    --max_new_tokens 512
```

**What this does:**
- Uses teacher model (Qwen-14B-Chat)
- Generates step-by-step reasoning for 377 questions
- Saves to `data/cot_dataset.jsonl`
- Uses 1 GPU (CUDA:0)
- Takes 10-15 minutes

**Verify:**
```bash
wc -l data/cot_dataset.jsonl  # Should show 377
head -n 1 data/cot_dataset.jsonl | python -m json.tool
```

---

### Stage 2: Train Student Model with LoRA

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

**What this does:**
- Uses student model (Qwen-7B)
- Trains with LoRA on CoT dataset
- Uses 4 GPUs (multi-GPU training)
- Saves to `results/student_cot_lora/`
- Takes 30-45 minutes

**Monitor progress:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Verify:**
```bash
ls results/student_cot_lora/
# Should see: adapter_config.json, adapter_model.bin, etc.
```

---

### Stage 3: Evaluate Trained Model

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --test_file data/test_questions.jsonl \
    --output_file results/predictions.jsonl \
    --max_new_tokens 512
```

**What this does:**
- Loads student model + LoRA weights
- Evaluates on 95 test questions
- Calculates accuracy
- Saves to `results/predictions.jsonl`
- Uses 1 GPU (CUDA:0)
- Takes 2-3 minutes

**View results:**
```bash
# View first prediction
head -n 1 results/predictions.jsonl | python -m json.tool

# Check accuracy (shown at end of script output)
tail -n 5 results/predictions.jsonl
```

---

## COMPLETE TIMELINE

```
PHASE 1 (Net Node):
├── Create environment: 2 min
├── Install dependencies: 5 min
└── Download models: 20-30 min
    TOTAL: ~35-40 minutes

PHASE 2 (GPU Node):
├── Stage 1 (CoT Generation): 10-15 min
├── Stage 2 (Training): 30-45 min
└── Stage 3 (Evaluation): 2-3 min
    TOTAL: ~45-65 minutes

GRAND TOTAL: ~1.5-2 hours
```

---

## VERIFICATION CHECKLIST

After completion, verify:

- [ ] `models/teacher/` exists and has config.json
- [ ] `models/student/` exists and has config.json
- [ ] `data/cot_dataset.jsonl` has 377 lines
- [ ] `results/student_cot_lora/` exists with adapter files
- [ ] `results/predictions.jsonl` has 95 lines
- [ ] Accuracy printed at end of evaluation

---

## QUICK COMMANDS

```bash
# Check everything exists
ls models/teacher/config.json models/student/config.json
wc -l data/*.jsonl
ls results/student_cot_lora/adapter_model.bin
wc -l results/predictions.jsonl

# View sample prediction
head -n 1 results/predictions.jsonl | python -m json.tool

# Check GPU usage
nvidia-smi
```

---

## TROUBLESHOOTING

**"No module named 'transformers'"**
→ Run: `conda activate cot_distill`

**"FileNotFoundError: models/teacher"**
→ Go back to Phase 1, download models on net node

**"CUDA out of memory"**
→ Reduce batch_size: `--batch_size 4` or `--batch_size 2`

**"Connection timeout" or "Cannot connect to HuggingFace"**
→ You're on GPU node without internet. Models must be downloaded on net node first.

**Training seems stuck**
→ Check `nvidia-smi` - GPUs should be at 90-100% utilization

---

## NEXT STEPS AFTER COMPLETION

1. **View predictions:**
   ```bash
   head -n 10 results/predictions.jsonl | python -m json.tool
   ```

2. **Analyze accuracy:**
   Check the accuracy percentage printed at end of evaluation

3. **Use trained model:**
   The LoRA weights in `results/student_cot_lora/` can be loaded with the student model for inference

4. **Scale up:**
   Generate more training data and retrain for better performance

---

## SUMMARY

**On Net Node (once):**
```bash
conda create -n cot_distill python=3.10 -y
conda activate cot_distill
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers peft datasets bitsandbytes accelerate tqdm sentencepiece protobuf deepspeed
cd /path/to/project
python scripts/download_models.py
```

**On GPU Node (every time):**
```bash
conda activate cot_distill
cd /path/to/project
./run_pipeline.sh
```

**Done!** 🎉
