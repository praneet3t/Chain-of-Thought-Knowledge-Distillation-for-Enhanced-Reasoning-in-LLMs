"""
Generate comprehensive math reasoning dataset.
Run this to create raw_questions.jsonl and test_questions.jsonl
"""

import json
import random
from pathlib import Path

random.seed(42)


def generate_arithmetic_questions(n=100):
    """Generate basic arithmetic questions."""
    questions = []
    
    # Addition
    for _ in range(n // 5):
        a, b = random.randint(100, 999), random.randint(100, 999)
        questions.append({
            "question": f"What is {a} + {b}?",
            "answer": str(a + b)
        })
    
    # Subtraction
    for _ in range(n // 5):
        a, b = random.randint(500, 999), random.randint(100, 499)
        questions.append({
            "question": f"What is {a} - {b}?",
            "answer": str(a - b)
        })
    
    # Multiplication
    for _ in range(n // 5):
        a, b = random.randint(10, 99), random.randint(10, 99)
        questions.append({
            "question": f"Calculate {a} × {b}.",
            "answer": str(a * b)
        })
    
    # Division
    for _ in range(n // 5):
        b = random.randint(10, 50)
        a = b * random.randint(10, 50)
        questions.append({
            "question": f"What is {a} ÷ {b}?",
            "answer": str(a // b)
        })
    
    # Mixed operations
    for _ in range(n // 5):
        a, b, c = random.randint(10, 50), random.randint(10, 50), random.randint(5, 20)
        result = a + b * c
        questions.append({
            "question": f"Calculate {a} + {b} × {c}.",
            "answer": str(result)
        })
    
    return questions


def generate_percentage_questions(n=80):
    """Generate percentage calculation questions."""
    questions = []
    
    for _ in range(n):
        percentage = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
        number = random.randint(50, 500)
        answer = (percentage * number) // 100
        
        questions.append({
            "question": f"What is {percentage}% of {number}?",
            "answer": str(answer)
        })
    
    return questions


def generate_algebra_questions(n=80):
    """Generate simple algebra questions."""
    questions = []
    
    # Linear equations: ax + b = c
    for _ in range(n // 2):
        a = random.randint(2, 10)
        b = random.randint(-20, 20)
        x = random.randint(1, 20)
        c = a * x + b
        
        if b >= 0:
            questions.append({
                "question": f"Solve for x: {a}x + {b} = {c}",
                "answer": str(x)
            })
        else:
            questions.append({
                "question": f"Solve for x: {a}x - {abs(b)} = {c}",
                "answer": str(x)
            })
    
    # Linear equations: ax - b = c
    for _ in range(n // 2):
        a = random.randint(2, 10)
        b = random.randint(1, 20)
        x = random.randint(1, 20)
        c = a * x - b
        
        questions.append({
            "question": f"Solve for x: {a}x - {b} = {c}",
            "answer": str(x)
        })
    
    return questions


def generate_word_problems(n=120):
    """Generate word problems."""
    questions = []
    
    # Speed/distance/time
    for _ in range(n // 4):
        distance = random.randint(50, 500)
        time = random.randint(2, 10)
        speed = distance // time
        
        questions.append({
            "question": f"A car travels {distance} km in {time} hours. What is its average speed in km/h?",
            "answer": f"{speed} km/h"
        })
    
    # Area problems
    for _ in range(n // 4):
        length = random.randint(5, 30)
        width = random.randint(5, 30)
        area = length * width
        
        questions.append({
            "question": f"A rectangle has a length of {length} cm and a width of {width} cm. What is its area in square cm?",
            "answer": str(area)
        })
    
    # Price calculations
    for _ in range(n // 4):
        items = random.randint(3, 12)
        price_per_item = random.uniform(1.5, 9.9)
        total = items * price_per_item
        
        questions.append({
            "question": f"If one apple costs ${price_per_item:.2f}, how much do {items} apples cost?",
            "answer": f"${total:.2f}"
        })
    
    # Age problems
    for _ in range(n // 4):
        current_age = random.randint(10, 50)
        years_ago = random.randint(5, 20)
        past_age = current_age - years_ago
        
        questions.append({
            "question": f"John is {current_age} years old now. How old was he {years_ago} years ago?",
            "answer": str(past_age)
        })
    
    return questions


def generate_fraction_questions(n=60):
    """Generate fraction questions."""
    questions = []
    
    for _ in range(n // 2):
        numerator = random.randint(1, 10)
        denominator = random.randint(2, 12)
        whole = random.randint(10, 100)
        result = (numerator * whole) // denominator
        
        questions.append({
            "question": f"What is {numerator}/{denominator} of {whole}?",
            "answer": str(result)
        })
    
    for _ in range(n // 2):
        a_num, a_den = random.randint(1, 5), random.randint(2, 8)
        b_num, b_den = random.randint(1, 5), random.randint(2, 8)
        
        # Same denominator for simplicity
        if a_den == b_den:
            result_num = a_num + b_num
            questions.append({
                "question": f"Add the fractions: {a_num}/{a_den} + {b_num}/{b_den}",
                "answer": f"{result_num}/{a_den}"
            })
    
    return questions


def generate_ratio_questions(n=60):
    """Generate ratio and proportion questions."""
    questions = []
    
    for _ in range(n):
        ratio_a = random.randint(2, 8)
        ratio_b = random.randint(2, 8)
        multiplier = random.randint(5, 20)
        total = (ratio_a + ratio_b) * multiplier
        part_a = ratio_a * multiplier
        
        questions.append({
            "question": f"Two numbers are in the ratio {ratio_a}:{ratio_b}. If their sum is {total}, what is the larger number?",
            "answer": str(max(part_a, (ratio_b * multiplier)))
        })
    
    return questions


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    print("Generating comprehensive math dataset...")
    
    # Generate all questions
    all_questions = []
    all_questions.extend(generate_arithmetic_questions(100))
    all_questions.extend(generate_percentage_questions(80))
    all_questions.extend(generate_algebra_questions(80))
    all_questions.extend(generate_word_problems(120))
    all_questions.extend(generate_fraction_questions(60))
    all_questions.extend(generate_ratio_questions(60))
    
    # Shuffle
    random.shuffle(all_questions)
    
    # Ensure we have enough questions
    print(f"Total questions generated: {len(all_questions)}")
    
    # Split into train (first 80%) and test (last 20%)
    split_idx = int(len(all_questions) * 0.8)
    train_questions = all_questions[:split_idx]
    test_questions = all_questions[split_idx:split_idx+100]  # Take 100 for test
    
    # Save training questions (without answers for CoT generation)
    train_file = data_dir / "raw_questions.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_questions:
            f.write(json.dumps({"question": item["question"]}, ensure_ascii=False) + '\n')
    
    print(f"[OK] Saved {len(train_questions)} training questions to {train_file}")
    
    # Save test questions (with answers for evaluation)
    test_file = data_dir / "test_questions.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_questions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"[OK] Saved {len(test_questions)} test questions to {test_file}")
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"\nTraining set: {len(train_questions)} questions")
    print(f"Test set: {len(test_questions)} questions")
    print("\nCategories:")
    print("  - Arithmetic: 100 questions")
    print("  - Percentages: 80 questions")
    print("  - Algebra: 80 questions")
    print("  - Word Problems: 120 questions")
    print("  - Fractions: 60 questions")
    print("  - Ratios: 60 questions")
    print("\nNext step: Run gen_cot.py to generate reasoning traces")


if __name__ == "__main__":
    main()
