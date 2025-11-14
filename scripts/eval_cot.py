# scripts/eval_cot.py
"""
Evaluate a base model with optional LoRA adapter (PEFT).
Supports local_files_only so it runs on an offline GPU node.
Outputs JSONL with fields: question, generated, pred, gold (if provided).
"""

import argparse, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from scripts.helpers import extract_final_answer

PROMPT_PREFIX = "### Question:\n{question}\n\n### Answer:\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="path to base student model (local)")
    parser.add_argument("--lora_model", default=None, help="path to LoRA adapter dir (local)")
    parser.add_argument("--input", default="data/test_questions.jsonl")
    parser.add_argument("--output", default="results/predictions.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", local_files_only=True)
    model.eval()

    if args.lora_model:
        print("Applying PEFT adapter from", args.lora_model)
        model = PeftModel.from_pretrained(model, args.lora_model, local_files_only=True)

    fout = open(args.output, "w", encoding="utf-8")
    total = 0
    correct = 0
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            question = obj["question"]
            gold = obj.get("answer", None)
            prompt = PROMPT_PREFIX.format(question=question)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=0.0, do_sample=False)
            out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            generated = out_text[len(prompt):] if out_text.startswith(prompt) else out_text
            pred = extract_final_answer(generated)
            out_record = {"question": question, "generated": generated, "pred": pred}
            if gold is not None:
                out_record["gold"] = gold
                if str(pred).strip() == str(gold).strip():
                    correct += 1
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            total += 1

    fout.close()
    if total:
        print(f"Saved outputs to {args.output}")
        print("Accuracy:", correct, "/", total, correct/total)

if __name__ == "__main__":
    main()
