"""
STT module — OpenAI Whisper speech-to-text.
Supports audio bytes from streamlit-webrtc or uploaded files.
"""
import io
import os
import tempfile
import numpy as np

_model = None
_model_size = "base"   # change to "tiny" for faster CPU inference


def get_model():
    """Lazy-load Whisper model (singleton)."""
    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model(_model_size)
    return _model


def transcribe_bytes(audio_bytes: bytes, language: str | None = None) -> str:
    """
    Transcribe raw audio bytes (WAV or WebM/opus from browser).
    Returns the transcribed string, or empty string on failure.
    """
    try:
        model = get_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        options = {}
        if language:
            options["language"] = language

        result = model.transcribe(tmp_path, **options)
        os.unlink(tmp_path)
        return result["text"].strip()

    except Exception as e:
        print(f"[STT] Transcription error: {e}")
        return ""


def transcribe_file(file_path: str, language: str | None = None) -> str:
    """Transcribe a file on disk."""
    try:
        model = get_model()
        options = {}
        if language:
            options["language"] = language
        result = model.transcribe(file_path, **options)
        return result["text"].strip()
    except Exception as e:
        print(f"[STT] File transcription error: {e}")
        return ""


def whisper_available() -> bool:
    try:
        import whisper
        return True
    except ImportError:
        return False
