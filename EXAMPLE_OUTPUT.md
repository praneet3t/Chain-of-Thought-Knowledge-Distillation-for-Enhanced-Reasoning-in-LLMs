# Example Chain-of-Thought Output

This document shows what the generated CoT reasoning should look like at each stage.

## Stage 1: Generated CoT Dataset (cot_dataset.jsonl)

After running `gen_cot.py`, your `data/cot_dataset.jsonl` should contain entries like:

```json
{
  "question": "What is 15% of 240?",
  "cot": "To find 15% of 240, I need to multiply 240 by 0.15.\n\nStep 1: Convert percentage to decimal\n15% = 15/100 = 0.15\n\nStep 2: Multiply\n240 × 0.15 = 36\n\nFinal Answer: 36"
}
```

```json
{
  "question": "If a train travels 120 km in 2 hours, what is its average speed?",
  "cot": "To find average speed, I use the formula: speed = distance / time\n\nGiven:\n- Distance = 120 km\n- Time = 2 hours\n\nStep 1: Apply the formula\nSpeed = 120 km / 2 hours\n\nStep 2: Calculate\nSpeed = 60 km/h\n\nFinal Answer: 60 km/h"
}
```

## Stage 2: Training Format

During training, the data is formatted as:

```
### Question:
What is 15% of 240?

### Answer:
To find 15% of 240, I need to multiply 240 by 0.15.

Step 1: Convert percentage to decimal
15% = 15/100 = 0.15

Step 2: Multiply
240 × 0.15 = 36

Final Answer: 36
```

## Stage 3: Evaluation Output (predictions.jsonl)

After running `eval_cot.py`, your `results/predictions.jsonl` should contain:

```json
{
  "question": "What is 20% of 150?",
  "generated_text": "To find 20% of 150, I need to multiply 150 by 0.20.\n\nStep 1: Convert percentage to decimal\n20% = 20/100 = 0.20\n\nStep 2: Multiply\n150 × 0.20 = 30\n\nFinal Answer: 30",
  "prediction": "30",
  "gold_answer": "30"
}
```

## Key Points

1. **CoT Structure**: The teacher model generates step-by-step reasoning that breaks down the problem
2. **Final Answer Format**: Always ends with "Final Answer: [answer]" for easy extraction
3. **Training Format**: Uses clear delimiters (### Question: / ### Answer:) for the student model
4. **Prediction Extraction**: The `extract_final_answer()` function pulls out just the final answer for accuracy calculation

## Quality Indicators

Good CoT reasoning should:
- Show clear logical steps
- Explain intermediate calculations
- Use consistent formatting
- End with explicit "Final Answer:" statement
- Be concise but complete

Poor CoT reasoning might:
- Jump to conclusions without steps
- Have inconsistent formatting
- Miss the "Final Answer:" marker
- Be overly verbose or repetitive
