# 🚀 START HERE

## Welcome to Chain-of-Thought Distillation Project

This is a **production-ready** project for training small language models to perform step-by-step reasoning using knowledge distillation from larger teacher models.

---

## ⚡ Quick Start (2 Commands)

### 1️⃣ On Net Node (with internet) - ONE TIME
```bash
conda create -n cot_distill python=3.10 -y && conda activate cot_distill
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers peft datasets bitsandbytes accelerate tqdm sentencepiece protobuf deepspeed
cd /path/to/chain-of-thought-qwen
python scripts/download_models.py
```

### 2️⃣ On GPU Node (offline) - RUN TRAINING
```bash
conda activate cot_distill
cd /path/to/chain-of-thought-qwen
./run_pipeline.sh
```

**Total Time: ~1.5 hours**

---

## 📚 Documentation Guide

Choose your path:

### 🎯 **I want to run it NOW**
→ Read: **ORDER_OF_EXECUTION.md**
- Step-by-step commands
- Copy-paste ready
- Verification steps

### 🔧 **I need HPC cluster setup**
→ Read: **HPC_SETUP.md**
- Offline mode setup
- Multi-GPU configuration
- Shared filesystem guide

### ⚡ **I want a quick reference**
→ Read: **QUICK_REFERENCE.md**
- All commands in one page
- Monitoring tips
- Troubleshooting table

### 📖 **I want complete documentation**
→ Read: **README.md**
- Full project details
- Hyperparameter guide
- Troubleshooting section

### 🎓 **I want to understand the project**
→ Read: **FINAL_SUMMARY.md**
- Project overview
- Technical specifications
- Expected results

---

## 📊 What You Get

- **Dataset**: 377 training + 95 test math questions
- **Models**: Qwen-14B (teacher) + Qwen-7B (student)
- **Method**: LoRA-based distillation
- **Output**: Trained model with 70-85% accuracy
- **Size**: LoRA adapters only ~50MB

---

## 🎯 Project Structure

```
chain-of-thought-qwen/
├── 📁 configs/          # Multi-GPU training config
├── 📁 data/             # 377 train + 95 test questions
├── 📁 models/           # Teacher & student models (download first!)
├── 📁 scripts/          # All Python scripts
├── 📁 results/          # Training outputs
├── 🚀 run_pipeline.sh   # Automated execution
└── 📚 Documentation files
```

---

## ✅ Prerequisites

- **Hardware**: 4x A100 GPUs (or 1x A100 40GB minimum)
- **Storage**: ~50GB free space
- **Network**: Offline GPU node + Online net node
- **Software**: Conda, CUDA 11.8+

---

## 🔄 Execution Flow

```
Net Node (Internet):
  └─> Download models (20-30 min)

GPU Node (Offline):
  ├─> Generate CoT (10-15 min, 1 GPU)
  ├─> Train Student (30-45 min, 4 GPUs)
  └─> Evaluate (2-3 min, 1 GPU)
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| **START_HERE.md** | 👈 You are here |
| **ORDER_OF_EXECUTION.md** | Exact execution order |
| **QUICK_REFERENCE.md** | All commands, one page |
| **HPC_SETUP.md** | Cluster-specific setup |
| **FINAL_SUMMARY.md** | Complete overview |
| **run_pipeline.sh** | Automated script |

---

## 🎬 What Happens When You Run

### Stage 1: Generate CoT Dataset
- Teacher model reads 377 questions
- Generates step-by-step reasoning
- Saves to `data/cot_dataset.jsonl`

### Stage 2: Train Student Model
- Student model learns from CoT examples
- Uses LoRA (efficient fine-tuning)
- Saves weights to `results/student_cot_lora/`

### Stage 3: Evaluate
- Tests on 95 held-out questions
- Calculates accuracy
- Saves predictions to `results/predictions.jsonl`

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| No internet on GPU node | Download models on net node first |
| CUDA out of memory | Reduce `--batch_size` to 4 or 2 |
| Module not found | Activate conda environment |
| Models not found | Run `download_models.py` on net node |

---

## 📞 Need Help?

1. **Quick issue?** → Check **QUICK_REFERENCE.md** troubleshooting table
2. **Setup problem?** → Read **HPC_SETUP.md** section 3
3. **Execution error?** → Follow **ORDER_OF_EXECUTION.md** step-by-step
4. **Understanding?** → Read **FINAL_SUMMARY.md**

---

## 🎯 Next Steps

1. **Read**: ORDER_OF_EXECUTION.md
2. **Setup**: Download models on net node
3. **Run**: Execute pipeline on GPU node
4. **Verify**: Check results in `results/predictions.jsonl`

---

## 📈 Expected Results

After successful execution:
- ✅ Trained LoRA model (~50MB)
- ✅ 95 predictions with accuracy score
- ✅ 70-85% accuracy on math problems
- ✅ Student model can do step-by-step reasoning

---

## 🚀 Ready to Start?

**→ Go to ORDER_OF_EXECUTION.md and follow the steps!**

---

## 📝 Quick Command Summary

```bash
# On net node (once)
python scripts/download_models.py

# On GPU node (every time)
./run_pipeline.sh

# That's it!
```

---

**Good luck! 🎉**
