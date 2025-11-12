import re
from typing import Optional


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from generated chain-of-thought text.
    
    Args:
        text: Generated text containing reasoning and final answer
        
    Returns:
        Cleaned final answer string
    """
    # Try to find "Final Answer:" pattern (case-insensitive)
    pattern = r'final\s+answer\s*:\s*(.+?)(?:\n|$)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        # Get first line after "Final Answer:"
        answer = match.group(1).strip()
        first_line = answer.split('\n')[0].strip()
        return first_line
    
    # Fallback: return last non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[-1] if lines else ""
