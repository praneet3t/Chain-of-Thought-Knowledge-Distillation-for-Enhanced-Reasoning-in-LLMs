# scripts/helpers.py
import re
from typing import Optional

def extract_final_answer(generated_text: str) -> str:
    """
    Extract final answer from generation. Looks for 'Final Answer:' (case-insensitive).
    If not found, returns the last non-empty line.
    """
    if generated_text is None:
        return ""
    s = generated_text.strip()
    # common markers
    markers = [r"Final Answer[:\-\s]*", r"Answer[:\-\s]*", r"Answer\s*\(final\)[:\-\s]*"]
    for m in markers:
        match = re.search(m + r"(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
        if match:
            ans = match.group(1).strip()
            return ans.splitlines()[0].strip()
    # fallback: last non-empty line
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    return lines[-1] if lines else s
