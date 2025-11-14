# scripts/train_cot_lora.py
"""
Train student with LoRA. Designed for Accelerate multi-GPU.
Loads student model (4-bit via bitsandbytes if available) and applies PEFT LoRA.
Saves checkpoints to --out_dir. Supports --resume_from_checkpoint.
"""

import argparse, os
from pathlib import Path
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math

def make_example(example, tokenizer, max_length=1024):
    q = example["question"].strip()
    cot = example["cot"].strip()
    text = f"### Question:\n{q}\n\n### Answer:\n{cot}\n"
    enc = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", default="models/student")
    parser.add_argument("--data", default="data/cot_dataset.jsonl")
    parser.add_argument("--out_dir", default="results/student_cot_lora")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer (local files only)...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_dir, use_fast=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading student model (4-bit if possible)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.student_dir,
        load_in_4bit=True,
        device_map="auto",
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PEFT LoRA applied. Trainable params:", trainable)

    print("Loading dataset...")
    ds = load_dataset("json", data_files=args.data)["train"]
    ds = ds.map(lambda ex: make_example(ex, tokenizer, max_length=args.max_length), remove_columns=ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("Training finished. Saving...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Saved to", out_dir)

if __name__ == "__main__":
    main()
