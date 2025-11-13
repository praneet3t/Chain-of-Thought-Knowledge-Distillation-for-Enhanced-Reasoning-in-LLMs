import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_PREFIX = "You are a careful reasoning assistant. Solve the problem step-by-step and end with a line that starts with 'Final Answer:'.\n\nQ: {question}\n\nA:"


def generate_cot(
    teacher_model_path: str,
    input_file: str,
    output_file: str,
    max_new_tokens: int = 512
):
    """Generate chain-of-thought reasoning using teacher model."""
    
    print(f"Loading teacher model from {teacher_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()
    
    # Load input questions
    print(f"Loading questions from {input_file}...")
    questions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))
    
    # Generate CoT for each question
    results = []
    print(f"Generating chain-of-thought reasoning for {len(questions)} questions...")
    
    for item in tqdm(questions):
        question = item['question']
        prompt = PROMPT_PREFIX.format(question=question)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated answer part (remove prompt)
        cot = generated[len(prompt):].strip()
        
        results.append({
            'question': question,
            'cot': cot,
            **{k: v for k, v in item.items() if k != 'question'}
        })
    
    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ Generated {len(results)} chain-of-thought examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CoT reasoning using teacher model")
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to teacher model directory")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with questions")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file for CoT dataset")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    generate_cot(
        teacher_model_path=args.teacher_model,
        input_file=args.input_file,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens
    )
