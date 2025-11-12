import argparse
import json
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


def format_training_example(question: str, cot: str) -> str:
    """Format question and CoT into training format."""
    return f"### Question:\n{question}\n\n### Answer:\n{cot}\n"


def load_cot_dataset(file_path: str):
    """Load CoT dataset from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def train_student_model(
    student_model_path: str,
    cot_dataset_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """Train student model with LoRA on CoT dataset."""
    
    print(f"Loading student model from {student_model_path}...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        student_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Set pad_token to eos_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and format dataset
    print(f"Loading CoT dataset from {cot_dataset_path}...")
    cot_data = load_cot_dataset(cot_dataset_path)
    
    formatted_texts = [
        format_training_example(item['question'], item['cot'])
        for item in cot_data
    ]
    
    # Tokenize dataset
    def tokenize_function(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    dataset = Dataset.from_dict({'text': formatted_texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit"
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student model with LoRA on CoT dataset")
    parser.add_argument("--student_model", type=str, required=True, help="Path to student model directory")
    parser.add_argument("--cot_dataset", type=str, required=True, help="Path to CoT dataset JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train_student_model(
        student_model_path=args.student_model,
        cot_dataset_path=args.cot_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
