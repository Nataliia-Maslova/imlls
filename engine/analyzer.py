"""
Phrase Analyzer — Step 8: Generative phrase evaluation.

Fixes vs previous version:
  1. Lesson phrases split on " - " so "Q - A" pairs are compared
     separately (user phrases are compared to question OR answer part)
  2. Relative calibration: score is normalized against the min/max
     similarity range within the lesson, so structurally correct but
     lexically novel phrases score realistically (70-85% not 34%)
  3. POS structure analysis unchanged (English only, nltk)
  4. Semantic analysis via MiniLM (multilingual)
"""
from __future__ import annotations
import re
import numpy as np

# ── Lazy singletons ───────────────────────────────────────────────────────
_st_model   = None
_nltk_ready = False


def _load_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _st_model


def _ensure_nltk():
    global _nltk_ready
    if not _nltk_ready:
        import nltk
        for pkg in ["punkt", "averaged_perceptron_tagger",
                    "punkt_tab", "averaged_perceptron_tagger_eng"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        _nltk_ready = True


# ═══════════════════════════════════════════════════════════════════════════
# Fix 1: split "Q - A" phrases into parts
# ═══════════════════════════════════════════════════════════════════════════

def _split_phrases(lesson_phrases: list[str]) -> list[str]:
    """
    Expand "Question - Answer" phrases into separate parts.
    "¿Qué tipo de chaqueta tienes? - Tengo una chaqueta larga."
    → ["¿Qué tipo de chaqueta tienes?", "Tengo una chaqueta larga."]

    Simple single-part phrases are kept as-is.
    """
    parts = []
    for phrase in lesson_phrases:
        if " - " in phrase:
            for p in phrase.split(" - "):
                p = p.strip()
                if len(p) > 2:
                    parts.append(p)
        else:
            parts.append(phrase.strip())
    return parts


# ═══════════════════════════════════════════════════════════════════════════
# Fix 2: relative calibration
# ═══════════════════════════════════════════════════════════════════════════

def _calibrate(raw_score: float, all_scores: list[float]) -> float:
    """
    Blend relative calibration with an absolute ceiling.

    Step 1 — relative: normalize against lesson range so structural matches
    score realistically even with novel vocabulary.
    Step 2 — absolute cap: if raw score is very low, cap the final score
    so unrelated phrases never reach "excellent".

    Absolute ceilings (tuned for MiniLM multilingual):
      raw < 0.20  → max 0.30  (clearly unrelated)
      raw < 0.30  → max 0.55  (weak connection)
      raw < 0.40  → max 0.75  (moderate connection)
      raw >= 0.40 → no cap
    """
    if not all_scores:
        return raw_score

    lo = min(all_scores)
    hi = max(all_scores)

    if hi - lo < 0.05:
        relative = raw_score
    else:
        relative = (raw_score - lo) / (hi - lo)
        relative = float(max(0.0, min(1.0, relative)))

    if raw_score < 0.20:
        ceiling = 0.30
    elif raw_score < 0.30:
        ceiling = 0.55
    elif raw_score < 0.40:
        ceiling = 0.75
    else:
        ceiling = 1.0

    return round(min(relative, ceiling), 4)


# ═══════════════════════════════════════════════════════════════════════════
# POS structure (English only)
# ═══════════════════════════════════════════════════════════════════════════

TAG_MAP = {
    "DT": "DET", "WDT": "DET",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB",
    "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB", "MD": "VERB",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON",
    "IN": "PREP",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
}


def _pos_pattern(text: str) -> list[str]:
    _ensure_nltk()
    import nltk
    try:
        tokens = nltk.word_tokenize(text.lower())
        tags   = nltk.pos_tag(tokens)
        return [TAG_MAP.get(t, "OTHER") for w, t in tags if re.match(r'\w', w)]
    except Exception:
        return []


def structure_score(user_phrase: str, lesson_phrases: list[str]) -> dict:
    from rapidfuzz import fuzz
    parts    = _split_phrases(lesson_phrases)
    user_pat = _pos_pattern(user_phrase)
    if not user_pat:
        return {"score": 0.0, "best_match": "", "user_pattern": [], "match_pattern": []}

    best_score, best_phrase, best_pat = 0.0, "", []
    all_sc = []
    for phrase in parts:
        pat  = _pos_pattern(phrase)
        sc   = fuzz.ratio(" ".join(user_pat), " ".join(pat)) / 100.0
        all_sc.append(sc)
        if sc > best_score:
            best_score, best_phrase, best_pat = sc, phrase, pat

    calibrated = _calibrate(best_score, all_sc)
    return {
        "score":         round(calibrated, 4),
        "raw_score":     round(best_score, 4),
        "best_match":    best_phrase,
        "user_pattern":  user_pat,
        "match_pattern": best_pat,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Semantic similarity (multilingual)
# ═══════════════════════════════════════════════════════════════════════════

def semantic_score(user_phrase: str, lesson_phrases: list[str]) -> dict:
    model = _load_st_model()
    parts = _split_phrases(lesson_phrases)   # Fix 1: use split parts

    vecs  = model.encode(parts, normalize_embeddings=True)
    u_vec = model.encode([user_phrase], normalize_embeddings=True)[0]

    raw_scores = [float(np.dot(u_vec, v)) for v in vecs]
    best_idx   = int(np.argmax(raw_scores))
    best_raw   = raw_scores[best_idx]

    # Fix 2: calibrate against lesson range
    calibrated = _calibrate(best_raw, raw_scores)

    # Also score vs centroid (for reference)
    centroid       = vecs.mean(axis=0)
    centroid      /= np.linalg.norm(centroid)
    centroid_score = float(np.dot(u_vec, centroid))

    return {
        "score":           round(calibrated, 4),       # calibrated — shown to user
        "raw_score":       round(best_raw, 4),          # raw cosine — for logging
        "centroid_score":  round(max(centroid_score, 0.0), 4),
        "best_match":      parts[best_idx],
        "all_scores":      [round(s, 4) for s in raw_scores],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Combined analysis
# ═══════════════════════════════════════════════════════════════════════════

def _vocab_penalty(user_phrase: str, lesson_phrases: list[str]) -> float:
    """
    Returns a penalty multiplier [0.6, 1.0] based on how many words
    in the user phrase appear in the lesson vocabulary.

    If most words are completely unknown → penalty up to 0.40 reduction.
    This prevents random sentences from scoring high just by sentence
    structure similarity.

    Logic:
      - Extract all unique words from lesson phrases (lowercase, no punct)
      - Count what fraction of user words appear in lesson vocab
      - coverage >= 0.5 → no penalty (1.0)
      - coverage 0.2–0.5 → mild penalty (0.8–1.0)
      - coverage < 0.2  → strong penalty (0.6)
    """
    import re
    lesson_words = set()
    for p in lesson_phrases:
        lesson_words.update(re.findall(r'[a-záéíóúüñàèìòùçäöü]+', p.lower()))

    user_words = re.findall(r'[a-záéíóúüñàèìòùçäöü]+', user_phrase.lower())
    if not user_words:
        return 1.0

    coverage = sum(1 for w in user_words if w in lesson_words) / len(user_words)

    if coverage >= 0.5:
        return 1.0
    elif coverage >= 0.2:
        # linear interpolation: 0.2→0.8, 0.5→1.0
        return round(0.8 + (coverage - 0.2) / 0.3 * 0.2, 3)
    else:
        return 0.6


def analyze_phrase(user_phrase: str,
                   lesson_phrases: list[str],
                   target_lang: str = "en",
                   known_words: list[str] | None = None) -> dict:
    """
    Full analysis combining structure (EN only) and semantic scores.
    Applies vocabulary penalty for words outside the lesson.
    """
    is_english = target_lang in ("en", "English")
    sem        = semantic_score(user_phrase, lesson_phrases)

    if is_english:
        struct   = structure_score(user_phrase, lesson_phrases)
        base     = round(0.4 * struct["score"] + 0.6 * sem["score"], 4)
    else:
        struct   = None
        base     = sem["score"]

    # Apply vocabulary penalty
    vocab_src = known_words if known_words else lesson_phrases
    penalty   = _vocab_penalty(user_phrase, vocab_src)
    combined  = round(base * penalty, 4)

    # Verdict thresholds
    if combined >= 0.80:
        verdict = "excellent"
    elif combined >= 0.60:
        verdict = "good"
    elif combined >= 0.40:
        verdict = "fair"
    else:
        verdict = "weak"

    return {
        "structure":    struct,
        "semantic":     sem,
        "combined":     combined,
        "vocab_penalty": penalty,
        "verdict":      verdict,
    }


def models_available() -> bool:
    try:
        import sentence_transformers  # noqa
        return True
    except ImportError:
        return False
