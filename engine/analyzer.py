"""
Phrase Analyzer — Step 8: Generative phrase evaluation.

Two independent scores:
  1. Structure Score  — POS pattern match via nltk (lightweight, ~3MB)
  2. Semantic Score   — cosine similarity via sentence-transformers MiniLM (~90MB)

Both models are lazy-loaded (only when step 8 is first used).
"""
from __future__ import annotations
import re
from functools import lru_cache

# ── Lazy model singletons ─────────────────────────────────────────────────
_st_model   = None   # sentence-transformers
_nltk_ready = False  # nltk punkt + averaged_perceptron_tagger


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
        for pkg in ["punkt", "averaged_perceptron_tagger", "punkt_tab",
                    "averaged_perceptron_tagger_eng"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        _nltk_ready = True


# ═══════════════════════════════════════════════════════════════════════════
# POS structure helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pos_pattern(text: str) -> list[str]:
    """
    Return simplified POS tag sequence for an English phrase.
    Tags collapsed to: DET, ADJ, NOUN, VERB, PRON, PREP, ADV, OTHER
    """
    _ensure_nltk()
    import nltk

    TAG_MAP = {
        "DT":  "DET",   "WDT": "DET",
        "JJ":  "ADJ",   "JJR": "ADJ",  "JJS": "ADJ",
        "NN":  "NOUN",  "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
        "VB":  "VERB",  "VBD": "VERB", "VBG": "VERB",
        "VBN": "VERB",  "VBP": "VERB", "VBZ": "VERB",
        "MD":  "VERB",
        "PRP": "PRON",  "PRP$": "PRON", "WP": "PRON",
        "IN":  "PREP",
        "RB":  "ADV",   "RBR": "ADV",  "RBS": "ADV",
    }

    try:
        tokens = nltk.word_tokenize(text.lower())
        tags   = nltk.pos_tag(tokens)
        return [TAG_MAP.get(t, "OTHER") for _, t in tags
                if re.match(r'\w', _)]
    except Exception:
        return []


def structure_score(user_phrase: str, lesson_phrases: list[str]) -> dict:
    """
    Compare POS pattern of user_phrase against all lesson phrase patterns.
    Returns best match score and the closest lesson phrase.
    """
    user_pat = _pos_pattern(user_phrase)
    if not user_pat:
        return {"score": 0.0, "best_match": "", "user_pattern": [], "match_pattern": []}

    from rapidfuzz import fuzz

    best_score  = 0.0
    best_phrase = ""
    best_pat    = []

    for phrase in lesson_phrases:
        pat   = _pos_pattern(phrase)
        # Compare tag sequences as strings
        u_str = " ".join(user_pat)
        p_str = " ".join(pat)
        sc    = fuzz.ratio(u_str, p_str) / 100.0
        if sc > best_score:
            best_score  = sc
            best_phrase = phrase
            best_pat    = pat

    return {
        "score":         round(best_score, 4),
        "best_match":    best_phrase,
        "user_pattern":  user_pat,
        "match_pattern": best_pat,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Semantic similarity helpers
# ═══════════════════════════════════════════════════════════════════════════

def semantic_score(user_phrase: str, lesson_phrases: list[str]) -> dict:
    """
    Encode user_phrase and lesson centroid with MiniLM.
    Returns cosine similarity and closest individual phrase.
    """
    import numpy as np

    model  = _load_st_model()
    vecs   = model.encode(lesson_phrases, normalize_embeddings=True)
    u_vec  = model.encode([user_phrase],  normalize_embeddings=True)[0]

    # Score vs centroid
    centroid       = vecs.mean(axis=0)
    centroid      /= np.linalg.norm(centroid)
    centroid_score = float(np.dot(u_vec, centroid))

    # Score vs each phrase individually → find closest
    scores         = [float(np.dot(u_vec, v)) for v in vecs]
    best_idx       = int(np.argmax(scores))

    return {
        "score":       round(max(centroid_score, 0.0), 4),
        "best_match":  lesson_phrases[best_idx],
        "best_score":  round(scores[best_idx], 4),
        "all_scores":  [round(s, 4) for s in scores],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Combined analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_phrase(user_phrase: str,
                   lesson_phrases: list[str],
                   target_lang: str = "en") -> dict:
    """
    Full analysis: structure + semantic scores.

    For non-English target languages only semantic score is returned
    (POS tagger is English-only).

    Returns:
        {
          "structure": {...} or None,
          "semantic":  {...},
          "combined":  0.0–1.0,
          "verdict":   "excellent" | "good" | "fair" | "weak",
        }
    """
    is_english = target_lang in ("en", "English")

    sem = semantic_score(user_phrase, lesson_phrases)

    if is_english:
        struct = structure_score(user_phrase, lesson_phrases)
        # Weighted: 40% structure, 60% semantic
        combined = round(0.4 * struct["score"] + 0.6 * sem["score"], 4)
    else:
        struct   = None
        combined = sem["score"]

    if combined >= 0.85:
        verdict = "excellent"
    elif combined >= 0.70:
        verdict = "good"
    elif combined >= 0.50:
        verdict = "fair"
    else:
        verdict = "weak"

    return {
        "structure": struct,
        "semantic":  sem,
        "combined":  combined,
        "verdict":   verdict,
    }


def models_available() -> bool:
    """Check if sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa
        return True
    except ImportError:
        return False
