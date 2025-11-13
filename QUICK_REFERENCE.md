# Quick Reference Card

## 🚀 Complete Execution (Copy-Paste Ready)

### On Net Node (With Internet) - ONE TIME SETUP
```bash
# Setup environment
conda create -n cot_distill python=3.10 -y && conda activate cot_distill

# Install dependencies
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers==4.36.0 peft==0.7.1 datasets==2.16.0 bitsandbytes==0.41.3 accelerate==0.25.0 tqdm==4.66.1 sentencepiece==0.1.99 protobuf==3.20.3 deepspeed==0.12.6

# Download models
cd /path/to/chain-of-thought-qwen
python scripts/download_models.py
```

### On GPU Node (Offline) - RUN PIPELINE
```bash
# Activate and run
conda activate cot_distill
cd /path/to/chain-of-thought-qwen
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 📊 Dataset Info

- **Training**: 377 questions (6 categories)
- **Test**: 95 questions with answers
- **Categories**: Arithmetic, Algebra, Percentages, Word Problems, Fractions, Ratios

---

## ⚙️ Manual Execution (Individual Stages)

```bash
# Stage 1: Generate CoT (10-15 min, 1 GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/gen_cot.py \
    --teacher_model models/teacher \
    --input_file data/raw_questions.jsonl \
    --output_file data/cot_dataset.jsonl

# Stage 2: Train (30-45 min, 4 GPUs)
accelerate launch --config_file configs/accelerate_config.yaml \
    scripts/train_cot_lora.py \
    --student_model models/student \
    --cot_dataset data/cot_dataset.jsonl \
    --output_dir results/student_cot_lora

# Stage 3: Evaluate (2-3 min, 1 GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_cot.py \
    --base_model models/student \
    --lora_model results/student_cot_lora \
    --test_file data/test_questions.jsonl \
    --output_file results/predictions.jsonl
```

---

## 🔍 Monitoring Commands

```bash
# GPU usage
watch -n 1 nvidia-smi

# View predictions
head -n 3 results/predictions.jsonl | python -m json.tool

# Check file sizes
du -sh models/teacher models/student results/student_cot_lora

# Count lines
wc -l data/*.jsonl
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| No internet on GPU node | Pre-download on net node first |
| CUDA OOM | Reduce `--batch_size` to 4 or 2 |
| Slow generation | Normal for teacher model |
| Module not found | Activate conda environment |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `run_pipeline.sh` | Automated execution |
| `EXECUTION_GUIDE.md` | Step-by-step instructions |
| `HPC_SETUP.md` | Cluster setup guide |
| `configs/accelerate_config.yaml` | 4-GPU configuration |

---

## ⏱️ Timeline

| Stage | Time | GPUs |
|-------|------|------|
| Model Download | 20-30 min | N/A |
| CoT Generation | 10-15 min | 1 |
| Training | 30-45 min | 4 |
| Evaluation | 2-3 min | 1 |
| **TOTAL** | **~1-1.5 hours** | |

---

## 📈 Expected Results

- **Accuracy**: 70-85%
- **LoRA Size**: ~50MB
- **Training Loss**: 2.5 → 0.5

---

## 🎯 Project Structure

```
chain-of-thought-qwen/
├── configs/accelerate_config.yaml
├── data/
│   ├── raw_questions.jsonl (377)
│   ├── cot_dataset.jsonl (generated)
│   └── test_questions.jsonl (95)
├── models/
│   ├── teacher/ (Qwen-14B-Chat)
│   └── student/ (Qwen-7B)
├── scripts/
│   ├── download_models.py
│   ├── gen_cot.py
│   ├── train_cot_lora.py
│   └── eval_cot.py
├── results/
│   ├── student_cot_lora/
│   └── predictions.jsonl
└── run_pipeline.sh
```

---

## 📚 Documentation

1. **QUICK_REFERENCE.md** ← You are here
2. **EXECUTION_GUIDE.md** - Detailed steps
3. **HPC_SETUP.md** - Cluster setup
4. **FINAL_SUMMARY.md** - Complete overview
5. **README.md** - Full documentation
