# scripts/generate_dataset.py
"""
Simple deterministic math dataset generator (creates raw_questions.jsonl and test_questions.jsonl).
You can replace this with your own dataset; included to make the repo runnable for tests.
"""

import json
from pathlib import Path
import random

def generate_arithmetic(n):
    out = []
    for i in range(n):
        a = random.randint(2, 200)
        b = random.randint(2, 200)
        out.append({"question": f"What is {a} + {b}?", "answer": str(a + b)})
    return out

def generate_percentages(n):
    out = []
    for i in range(n):
        base = random.randint(10, 500)
        pct = random.choice([5,10,12,15,20,25,30])
        out.append({"question": f"What is {pct}% of {base}?", "answer": str(int(base * pct / 100))})
    return out

def main():
    random.seed(42)
    project = Path(".")
    data = []
    data += generate_arithmetic(100)
    data += generate_percentages(80)
    # keep small for test/demo; expand as desired
    raw = [ {"question": d["question"]} for d in data ]
    test = data[:20]  # small held-out example

    project.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    with open("data/raw_questions.jsonl", "w", encoding="utf-8") as f:
        for q in raw:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open("data/test_questions.jsonl", "w", encoding="utf-8") as f:
        for q in test:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print("Wrote data/raw_questions.jsonl and data/test_questions.jsonl (samples).")

if __name__ == "__main__":
    main()
