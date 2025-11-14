# scripts/gen_cot.py
"""
Generate chain-of-thought traces using the teacher model.
Run on GPU node (or single GPU). Uses local_files_only=True so it won't try to go online.
Supports --start_index to resume from partial output.
"""

import argparse, json, os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

PROMPT_PREFIX = (
    "You are a careful reasoning assistant. Solve the problem step-by-step and end with a line that starts with 'Final Answer:'.\n\n"
    "Q: {question}\n\nA:"
)

def generate_for_question(model, tokenizer, question, max_new_tokens=256, temperature=0.2, device=None):
    prompt = PROMPT_PREFIX.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        top_p=1.0,
        repetition_penalty=1.03,
    )
    out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    out_only = out[len(prompt):].strip() if out.startswith(prompt) else out
    return out_only

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_dir", default="models/teacher")
    parser.add_argument("--input", default="data/raw_questions.jsonl")
    parser.add_argument("--output", default="data/cot_dataset.jsonl")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start_index", type=int, default=0, help="skip already-generated lines")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer & model (local files only)...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir, use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.teacher_dir, device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    model.eval()
    device = next(model.parameters()).device

    # read input
    with open(args.input, "r", encoding="utf-8") as fin:
        all_qs = [json.loads(line)["question"] for line in fin]

    total = len(all_qs) if args.num_samples is None else min(args.num_samples, len(all_qs))
    print(f"Total questions available: {len(all_qs)}, will process: {total}, start_index={args.start_index}")

    # skip already generated
    start = args.start_index
    pbar = tqdm(range(start, total))
    written = 0
    mode = "a" if start > 0 or out_path.exists() else "w"
    with open(args.output, mode, encoding="utf-8") as fout:
        for i in pbar:
            q = all_qs[i]
            pbar.set_description(f"Generating {i+1}/{total}")
            cot = generate_for_question(model, tokenizer, q, max_new_tokens=args.max_new_tokens, device=device)
            fout.write(json.dumps({"question": q, "cot": cot}, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
    print(f"Done. Generated {written} examples. Saved to {args.output}")

if __name__ == "__main__":
    main()
