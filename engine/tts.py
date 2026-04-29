"""
TTS module — Google TTS with local file caching.
Audio files are cached to avoid re-generating the same phrases.
"""
import hashlib
import os
from pathlib import Path

# pygame is NOT used on Streamlit Cloud — audio plays via base64 HTML in the browser
# It is optional for local development only
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

CACHE_DIR = Path(__file__).parent.parent / "audio_cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_audio_path(text: str, lang: str) -> str | None:
    """
    Return path to an MP3 file for the given text+lang.
    Generates and caches if not already on disk.
    Returns None if gTTS is not available.
    """
    if not GTTS_AVAILABLE:
        return None

    key  = hashlib.md5(f"{lang}::{text}".encode()).hexdigest()
    path = CACHE_DIR / f"{key}.mp3"

    if not path.exists():
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(path))
        except Exception as e:
            print(f"[TTS] Error generating audio: {e}")
            return None

    return str(path)


def tts_available() -> bool:
    return GTTS_AVAILABLE
