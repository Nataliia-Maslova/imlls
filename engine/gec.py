"""
engine/gec.py — Grammatical Error Correction
Fine-tuned T5-small on JFLEG. English only.

from pathlib import Path
import torch

_model = T5ForConditionalGeneration.from_pretrained("natashasms/imlls-gec")
_tokenizer = None
#MODEL_PATH = Path(__file__).parent.parent / "gec_model"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load():
    global _model, _tokenizer
    if _model is None:
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        _model     = T5ForConditionalGeneration.from_pretrained(str(MODEL_PATH))
        _model.to(DEVICE)
        _model.eval()
    return _model, _tokenizer"""

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

_model = None
_tokenizer = None

MODEL_NAME = "natashasms/imlls-gec"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load():
    global _model, _tokenizer

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        _model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

        _model.to(DEVICE)
        _model.eval()

    return _model, _tokenizer


def correct(text: str) -> str:
    """Return grammatically corrected version of text."""
    try:
        model, tok = _load()
        input_ids = tok(
            f"grammar: {text}",
            return_tensors="pt",
            max_length=128,
            truncation=True,
        ).input_ids.to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=128,
                                     num_beams=4, early_stopping=True)
        return tok.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[GEC] Error: {e}")
        return text


def gec_available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False