import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from helpers import extract_final_answer


def evaluate_model(
    base_model_path: str,
    lora_model_path: str,
    test_file: str,
    output_file: str,
    max_new_tokens: int = 512
):
    """Evaluate fine-tuned model on test questions."""
    
    print(f"Loading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA weights from {lora_model_path}...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    
    # Set pad_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test questions
    print(f"Loading test questions from {test_file}...")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    has_gold_answers = 'answer' in test_data[0]
    
    print(f"Evaluating on {len(test_data)} questions...")
    
    for item in tqdm(test_data):
        question = item['question']
        prompt = f"### Question:\n{question}\n\n### Answer:\n"
        
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
        # Extract only the generated answer part
        generated_text = generated[len(prompt):].strip()
        
        # Extract final answer
        prediction = extract_final_answer(generated_text)
        
        result = {
            'question': question,
            'generated_text': generated_text,
            'prediction': prediction
        }
        
        # Calculate accuracy if gold answers provided
        if has_gold_answers:
            gold_answer = str(item['answer']).strip()
            result['gold_answer'] = gold_answer
            
            if prediction.lower() == gold_answer.lower():
                correct += 1
            total += 1
        
        results.append(result)
    
    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Print accuracy if applicable
    if has_gold_answers:
        accuracy = correct / total * 100
        print(f"\n✓ Evaluation complete!")
        print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    else:
        print(f"\n✓ Evaluation complete! Generated predictions for {len(results)} questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned CoT model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base student model directory")
    parser.add_argument("--lora_model", type=str, required=True, help="Path to fine-tuned LoRA model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Test questions JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for predictions")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    evaluate_model(
        base_model_path=args.base_model,
        lora_model_path=args.lora_model,
        test_file=args.test_file,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens
    )
