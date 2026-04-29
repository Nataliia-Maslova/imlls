"""
Scorer — text normalization and similarity scoring using RapidFuzz.
"""
import re
from rapidfuzz import fuzz

PASS_THRESHOLD = 0.82   # 82% similarity = correct


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(user_input: str, expected: str) -> float:
    """Return similarity score 0.0–1.0."""
    a = normalize(user_input)
    b = normalize(expected)
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def evaluate(user_input: str, expected: str) -> dict:
    """Full evaluation result dict."""
    score = similarity(user_input, expected)
    return {
        "transcribed": user_input,
        "expected":    expected,
        "score":       round(score, 4),
        "passed":      score >= PASS_THRESHOLD,
    }
