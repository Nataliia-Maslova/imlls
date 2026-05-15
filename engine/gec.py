"""
engine/gec.py - Grammatical Error Correction
Fine-tuned T5 models hosted on HuggingFace Hub.
Supports English, Spanish and Korean.
"""

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

# Repo IDs of the fine-tuned models on HuggingFace Hub
MODELS = {
    "en": "natashasms/en-gec-model",
    "es": "natashasms/es-gec-model",
    "ko": "natashasms/ko-gec-model",
}

# Per-language cache: {lang: (model, tokenizer)}
_cache: dict = {}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(lang: str = "en"):
    """Lazy-load model + tokenizer for the requested language."""
    if lang not in MODELS:
        return None, None

    if lang not in _cache:
        repo_id = MODELS[lang]
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = T5ForConditionalGeneration.from_pretrained(repo_id)
        model.to(DEVICE)
        model.eval()
        _cache[lang] = (model, tokenizer)

    return _cache[lang]


def correct(text: str, lang: str = "en") -> str:
    """Return grammatically corrected version of `text` in the given language.

    Falls back to returning the original text on any error or for unsupported
    languages.
    """
    try:
        model, tokenizer = _load(lang)
        if model is None or tokenizer is None:
            return text

        input_ids = tokenizer(
            f"grammar: {text}",
            return_tensors="pt",
            max_length=128,
            truncation=True,
        ).input_ids.to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[GEC:{lang}] Error: {e}")
        return text


def gec_available(lang: str = "en") -> bool:
    """Check whether the GEC model for the given language can be loaded."""
    if lang not in MODELS:
        return False
    try:
        _load(lang)
        return True
    except Exception as e:
        print(f"[GEC:{lang}] Unavailable: {e}")
        return False


def supported_languages() -> list:
    """List of language codes that have a GEC model."""
    return list(MODELS.keys())
