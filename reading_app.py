"""
reading_app.py  —  IMLLS Reading Practice
==========================================
Запуск:  streamlit run reading_app.py
Дані:    data/reading_lessons.xlsx

Встановлення:
    pip install streamlit pandas openpyxl gtts edge-tts

Алгоритм (5 кроків):
    1. Послухай і повтори       — по одному слову з аудіо
    2. Прочитай слова           — читає сам, потім перевіряє аудіо
    3. Послухай і знайди        — слухає → вибирає зі списку
    4. Послухай і повтори       — ще раз всі слова по черзі
    5. Прочитай на час          — таймер + всі слова видно

Озвучка:
    - Уроки 1, 3, 8, 16 (букви):  phonemes/en/<letter>.mp3
    - Інші букви (Aa, Bb):        edge-tts SSML phoneme → fallback gTTS carrier
    - Слова (Bad, Man):           edge-tts / gTTS
"""

import asyncio
import base64
import hashlib
import random
import re
import time
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT      = Path(__file__).parent
sys.path.insert(0, str(ROOT))
CACHE_DIR = ROOT / "audio_cache" / "reading"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = ROOT / "data" / "reading_lessons.xlsx"

# Lessons that use pre-recorded phoneme audio from phonemes/en/<letter>.mp3
PHONEME_AUDIO_LESSONS = {1, 3, 8, 16}
PHONEMES_DIR    = ROOT / "phonemes" / "en"
PHONEMES_DIR_UK = ROOT / "phonemes" / "uk"
PHONEMES_DIR_KO = ROOT / "phonemes" / "ko"

# Spanish phoneme dispatch:
#   A E I O U Z Ñ H + all syllables → edge-tts Spanish voice (lowercase)
#   All consonants below → pre-recorded file from phonemes/en/
ES_PHONEME_MAP: dict[str, Path] = {
    # Special files
    "R":  PHONEMES_DIR / "rr.ogg",
    "J":  PHONEMES_DIR / "jota.ogg",
    "CH": PHONEMES_DIR / "ch.ogg",
    "LL": PHONEMES_DIR / "ll.ogg",
    "Y":  PHONEMES_DIR / "ll.ogg",   # Y sounds like LL in Spanish
    # Standard English phoneme mp3s
    "M":  PHONEMES_DIR / "m.mp3",
    "P":  PHONEMES_DIR / "p.mp3",
    "L":  PHONEMES_DIR / "l.mp3",
    "S":  PHONEMES_DIR / "s.mp3",
    "T":  PHONEMES_DIR / "t.mp3",
    "N":  PHONEMES_DIR / "n.mp3",
    "D":  PHONEMES_DIR / "d.mp3",
    "F":  PHONEMES_DIR / "f.mp3",
    "B":  PHONEMES_DIR / "b.mp3",
    "V":  PHONEMES_DIR / "v.mp3",
    "C":  PHONEMES_DIR / "c.mp3",
    "G":  PHONEMES_DIR / "g.mp3",
    "Q":  PHONEMES_DIR / "q.mp3",
    "X":  PHONEMES_DIR / "x.mp3",
    "K":  PHONEMES_DIR / "k.mp3",
    "W":  PHONEMES_DIR / "w.mp3",
}

# Korean consonant phoneme set — pre-recorded .ogg files in phonemes/ko/
KO_CONSONANTS = {
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ",
    "ㅈ", "ㅎ", "ㅋ", "ㅌ", "ㅍ", "ㅊ",
    "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ",
}

# ── Multi-language TTS / Whisper config ───────────────────────────────────
TTS_CONFIG = {
    "en": {"voice": "en-US-JennyNeural", "gtts": "en"},
    "uk": {"voice": "uk-UA-PolinaNeural", "gtts": "uk"},
    "es": {"voice": "es-ES-ElviraNeural", "gtts": "es"},
    "ko": {"voice": "ko-KR-SunHiNeural",  "gtts": "ko"},
}
LANG_LABELS = {
    "en": "English 🇬🇧",
    "uk": "Українська 🇺🇦",
    "es": "Español 🇪🇸",
    "ko": "한국어 🇰🇷",
}
WHISPER_LANG = {"en": "en", "uk": "uk", "es": "es", "ko": "ko"}

# Native language → column name in «Правила» sheet (rules for English lessons)
NATIVE_TO_RULES_COL = {
    "Ukrainian": "uk", "Russian": "ru",
    "English":   "en", "Spanish": "es",
}


def _r_lang() -> str:
    """Current target reading language from session state (default: 'en')."""
    return st.session_state.get("r_lang", "en")

# ── optional STT ──────────────────────────────────────────────────────────
try:
    from engine.stt import transcribe_bytes, whisper_available
    STT_OK = whisper_available()
except Exception:
    STT_OK = False

# ── optional similarity scorer ────────────────────────────────────────────
try:
    from engine.scorer import evaluate as _evaluate
    SCORER_OK = True
except Exception:
    SCORER_OK = False

# ── logging (CSV + Google Sheets via engine.logger) ───────────────────────
try:
    from engine.logger import SessionLogger, get_last_lesson, get_progress, save_progress
    LOGGER_OK = True
except Exception:
    LOGGER_OK = False


def _save_step_progress(lesson_id: int, step: int, user_id: str):
    """Persist current (lesson_id, step) so user can resume here next time.
    Saves at most once per (lesson_id, step) per session."""
    if not LOGGER_OK:
        return
    key = (lesson_id, step)
    if st.session_state.get("_r_last_saved_progress") == key:
        return
    try:
        save_progress(
            user_id               = user_id,
            language_pair         = _reading_lang_pair(),
            last_completed_lesson = int(lesson_id),
            last_step             = int(step),
        )
        st.session_state["_r_last_saved_progress"] = key
    except Exception as e:
        print(f"[reading_app] save_progress error: {e}")

def _reading_lang_pair() -> str:
    """Language pair key for progress logging, e.g. 'en-reading', 'uk-reading'."""
    return f"{_r_lang()}-reading"


def _get_logger():
    """Return (and lazily create) the SessionLogger for this reading session."""
    if not LOGGER_OK:
        return None
    if "r_logger" in st.session_state:
        return st.session_state["r_logger"]
    user_id = st.session_state.get("r_user", "anonymous")
    try:
        logger = SessionLogger(user_id, language_pair=_reading_lang_pair())
        st.session_state["r_logger"] = logger
        return logger
    except Exception as e:
        print(f"[reading_app] logger init failed: {e}")
        return None


def _log_score(step: int, phrase_id: int, similarity: float,
               response_time_ms: int, success: bool):
    """Log a score event for the current reading lesson."""
    logger = _get_logger()
    if logger is None:
        return
    try:
        logger.log(
            lesson_id=int(st.session_state.get("r_lesson", 0)),
            phrase_id=phrase_id,
            step=step,
            similarity=similarity,
            response_time_ms=response_time_ms,
            attempts=1,
            success=success,
            mode="reading",
        )
    except Exception as e:
        print(f"[reading_app] log error: {e}")


def score_audio(audio_bytes, expected_text, lang: str = None):
    """Transcribe via Whisper and score similarity vs expected_text."""
    if not STT_OK or not SCORER_OK or not audio_bytes:
        return None
    wh_lang = WHISPER_LANG.get(lang or _r_lang(), "en")
    try:
        text = transcribe_bytes(audio_bytes, language=wh_lang)
        return _evaluate(text, expected_text)
    except Exception as e:
        print(f"[score_audio] {e}")
        return None


def _audio_duration_ms(audio_bytes: bytes) -> int:
    """Audio duration in ms. Tries WAV header, falls back to size estimate."""
    if not audio_bytes:
        return 0
    try:
        import io, wave
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            return int(wf.getnframes() / wf.getframerate() * 1000)
    except Exception:
        pass
    # Fallback: ~16 kHz mono webm/opus ≈ 32 KB/s
    return max(0, int(len(audio_bytes) / 32000 * 1000))


PHONEME_WORD = {
    "æ":  "at", "e":  "egg", "ɪ":  "it", "ɔ":  "on", "ʌ":  "up",
    "ə":  "a", "ʊ":  "good",
    "i:": "see", "ɑ:": "far", "ɔ:": "or", "ɜ:": "her", "u:": "too",
    "eɪ": "say", "aɪ": "my", "ɔɪ": "boy", "aʊ": "now", "əʊ": "go",
    "ɪə": "here", "eə": "air", "ʊə": "pure",
    "b":  "buh", "d":  "duh", "f":  "fff", "g":  "guh", "h":  "huh",
    "j":  "yes", "k":  "kuh", "l":  "lll", "m":  "mmm", "n":  "nnn",
    "ŋ":  "ring", "p":  "puh", "r":  "rrr", "s":  "sss", "t":  "tuh",
    "v":  "vvv", "w":  "wet", "z":  "zzz",
    "ʒ":  "vision", "ʃ":  "shh", "tʃ": "church", "dʒ": "judge",
    "θ":  "thin", "ð":  "the", "ks": "fox", "kw": "quick",
}


def _cache_path(key_str: str, prefix: str = "a") -> Path:
    h = hashlib.md5(key_str.encode()).hexdigest()
    return CACHE_DIR / f"{prefix}_{h}.mp3"


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _gtts(text: str, path: Path, lang: str = "en"):
    from gtts import gTTS
    gtts_lang = TTS_CONFIG.get(lang, TTS_CONFIG["en"])["gtts"]
    GTts = gTTS(text=text, lang=gtts_lang, slow=True)
    GTts.save(str(path))


async def _edge(text: str, path: Path, rate: str = "-5%", voice: str = None):
    import edge_tts
    if voice is None:
        voice = TTS_CONFIG.get(_r_lang(), TTS_CONFIG["en"])["voice"]
    tts = edge_tts.Communicate(text, voice=voice, rate=rate)
    await tts.save(str(path))


async def _edge_ssml_phoneme(ipa: str, path: Path):
    """edge-tts SSML with IPA phoneme tag — exact sound."""
    import edge_tts
    ssml = (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xml:lang="en-US">'
        '<voice name="en-US-JennyNeural">'
        f'<prosody rate="-20%">'
        f'<phoneme alphabet="ipa" ph="{ipa}">a</phoneme>'
        '</prosody>'
        '</voice></speak>'
    )
    tts = edge_tts.Communicate(ssml, voice="en-US-JennyNeural")
    await tts.save(str(path))


def _edge_ok() -> bool:
    try:
        import edge_tts  # noqa
        return True
    except ImportError:
        return False


def _gtts_ok() -> bool:
    try:
        from gtts import gTTS  # noqa
        return True
    except ImportError:
        return False


def audio_for_word(word: str, lang: str = None):
    """Generate MP3 for a word or compound phrase. Cached permanently."""
    lang = lang or _r_lang()
    # Strip stress markers / curly apostrophes that TTS doesn’t handle well
    clean = word
    for ch in ("’", "’", "’", "`"):
        clean = clean.replace(ch, "")
    clean = clean.strip()
    path  = _cache_path(f"word::{lang}::{clean}", "w")
    if path.exists():
        return path
    voice = TTS_CONFIG.get(lang, TTS_CONFIG["en"])["voice"]
    try:
        if _edge_ok():
            _run_async(_edge(clean, path, voice=voice))
        elif _gtts_ok():
            _gtts(clean, path, lang=lang)
        else:
            return None
        return path if path.exists() else None
    except Exception as e:
        print(f"[audio_for_word] ‘{clean}’: {e}")
        return None


def audio_for_phoneme(ipa: str):
    """IPA phoneme → MP3. Edge-tts SSML first, then carrier word fallback."""
    ipa_clean = ipa.strip().strip("[]").strip()
    path      = _cache_path(f"phoneme::{ipa_clean}", "ph")
    if path.exists():
        return path

    if _edge_ok():
        try:
            _run_async(_edge_ssml_phoneme(ipa_clean, path))
            if path.exists() and path.stat().st_size > 2000:
                return path
            else:
                path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[phoneme SSML] '{ipa_clean}': {e}")
            path.unlink(missing_ok=True)

    carrier = PHONEME_WORD.get(ipa_clean)
    if not carrier:
        for k, v in PHONEME_WORD.items():
            if k in ipa_clean or ipa_clean.startswith(k):
                carrier = v
                break

    if carrier:
        try:
            if _edge_ok():
                _run_async(_edge(carrier, path, rate="-20%"))
            elif _gtts_ok():
                _gtts(carrier, path)
            return path if path.exists() else None
        except Exception as e:
            print(f"[phoneme carrier] '{carrier}': {e}")
    return None


def prerecorded_phoneme_path(word: str):
    """phonemes/en/<letter>.mp3 if it exists for the given word (e.g. 'Aa' → a.mp3)."""
    clean = word.strip()
    if not clean:
        return None
    first = clean[0].lower()
    if not first.isalpha():
        return None
    p = PHONEMES_DIR / f"{first}.mp3"
    return p if p.exists() else None


def audio_for_row(word: str, transcription: str, lesson_id=None, lang: str = None):
    """Smart dispatch per language:
    - English:    pre-recorded phonemes → IPA phoneme TTS → word TTS
    - Ukrainian:  phonemes/uk/<WORD>.ogg  → TTS lowercase fallback
    - Spanish:    ES_PHONEME_MAP for R/J/CH/LL → TTS lowercase for everything else
    - Korean:     phonemes/ko/<jamo>.ogg for consonants → TTS for vowels/syllables
    """
    lang = lang or _r_lang()

    w = word.strip()

    # ── Ukrainian ─────────────────────────────────────────────────────────
    if lang == "uk":
        # Try pre-recorded file: phonemes/uk/А.ogg, phonemes/uk/ДЖ.ogg, etc.
        p = PHONEMES_DIR_UK / f"{w}.ogg"
        if not p.exists():
            p = PHONEMES_DIR_UK / f"{w.upper()}.ogg"
        if p.exists():
            return p
        # Fallback: lowercase → edge-tts (syllables like МА → "ма")
        trans = str(transcription).strip()
        spoken_text = trans if (trans and trans.lower() != "nan") else w.lower()
        spoken = re.sub(r"\s*[–—‐‑‒\-]\s*", ", ", spoken_text)
        return audio_for_word(spoken, lang="uk")

    # ── Spanish ───────────────────────────────────────────────────────────
    if lang == "es":
        w_up = w.upper()
        # Specific consonants → pre-recorded phoneme file
        if w_up in ES_PHONEME_MAP:
            p = ES_PHONEME_MAP[w_up]
            if p.exists():
                return p
        # Vowels (A E I O U), Ñ, H and all syllables → edge-tts Spanish voice
        trans = str(transcription).strip()
        spoken_text = trans if (trans and trans.lower() != "nan") else w.lower()
        spoken = re.sub(r"\s*[–—‐‑‒\-]\s*", ", ", spoken_text)
        return audio_for_word(spoken, lang="es")

    # ── Korean ────────────────────────────────────────────────────────────────
    if lang == "ko":
        # Single consonant → pre-recorded ogg
        if w in KO_CONSONANTS:
            p = PHONEMES_DIR_KO / f"{w}.ogg"
            if p.exists():
                return p
        # Everything else (vowels, syllables, words) → edge-tts Korean voice
        trans = str(transcription).strip()
        spoken_text = trans if (trans and trans.lower() != "nan") else w
        spoken = re.sub(r"\s*[–—‐‑‒\-]\s*", ", ", spoken_text)
        return audio_for_word(spoken, lang="ko")

    # ── Other non-English ─────────────────────────────────────────────────────
    if lang != "en":
        trans = str(transcription).strip()
        spoken_text = trans if (trans and trans.lower() != "nan") else w.lower()
        spoken = re.sub(r"\s*[–—‐‑‒\-]\s*", ", ", spoken_text)
        return audio_for_word(spoken, lang=lang)

    # English path — original logic
    is_letter_row = bool(re.match(r"^[A-Za-z]{1,2}$", word.strip()))

    if lesson_id in PHONEME_AUDIO_LESSONS and is_letter_row:
        p = prerecorded_phoneme_path(word)
        if p:
            return p

    if is_letter_row:
        ipa = re.sub(r"[\[\]]", "", transcription).strip()
        return audio_for_phoneme(ipa)
    else:
        spoken = re.sub(r"\s*[–—‐‑‒\-]\s*", ", ", word.strip())
        return audio_for_word(spoken, lang="en")


def play(path, autoplay=False):
    """Render audio. Uses st.audio for correct refresh between reruns."""
    if not path or not Path(path).exists():
        st.caption("⚠️ Audio not available")
        return
    with open(path, "rb") as f:
        audio_bytes = f.read()
    try:
        st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay)
    except TypeError:
        # Streamlit < 1.34 doesn't support autoplay — fallback to HTML with unique nonce
        d = base64.b64encode(audio_bytes).decode()
        auto = "autoplay" if autoplay else ""
        nonce = hashlib.md5(str(path).encode()).hexdigest()[:8]
        st.markdown(
            f'<div data-audio-nonce="{nonce}">'
            f'<audio controls {auto} style="width:100%;border-radius:8px;margin:4px 0">'
            f'<source src="data:audio/mp3;base64,{d}" type="audio/mp3"></audio></div>',
            unsafe_allow_html=True,
        )



def autoplaylist_html(audio_paths, pause_secs=1.0, uid="pl"):
    """JS component: plays a list of MP3s sequentially with a fixed pause between."""
    import json as _json
    srcs = []
    for p in audio_paths:
        if p and Path(p).exists():
            with open(p, "rb") as f:
                srcs.append("data:audio/mp3;base64," + base64.b64encode(f.read()).decode())
        else:
            srcs.append("")
    srcs_js  = _json.dumps(srcs)
    pause_ms = int(pause_secs * 1000)
    n = len(srcs)
    return f"""
<div style="background:#FFFFFF;border:1px solid #E8E2D8;border-radius:12px;padding:14px 18px;margin:8px 0;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <button id="pl-btn-{uid}" onclick="plToggle_{uid}()"
      style="background:#ECEBFB;color:#4F46E5;border:1px solid #4F46E5;border-radius:8px;
             padding:7px 18px;cursor:pointer;font-family:JetBrains Mono,monospace;font-size:.88rem;">
      ▶ Play All
    </button>
    <span id="pl-stat-{uid}" style="color:#7A7390;font-size:.8rem;font-family:JetBrains Mono,monospace;">ready</span>
  </div>
  <div id="pl-bar-{uid}" style="margin-top:10px;display:flex;gap:4px;flex-wrap:wrap;"></div>
</div>
<script>
(function(){{
  const srcs={srcs_js}, pauseMs={pause_ms}, n={n}, uid='{uid}';
  let cur=-1, playing=false, aud=null, tmr=null;
  const bar=document.getElementById('pl-bar-'+uid);
  for(let i=0;i<n;i++){{
    const d=document.createElement('div'); d.id='dot-'+uid+'-'+i;
    d.style.cssText='width:10px;height:10px;border-radius:50%;background:#ECEBFB;transition:.2s;';
    bar.appendChild(d);
  }}
  function dot(i,c){{
    const d=document.getElementById('dot-'+uid+'-'+i); if(!d) return;
    d.style.background = c==='active' ? '#4F46E5' : c==='done' ? '#1FB888' : '#ECEBFB';
  }}
  function ensureAud(){{
    // Create ONE Audio element only inside a user-gesture handler.
    // iOS Safari blocks new Audio()/play() called from setTimeout because
    // they lose the gesture. Reusing one element keeps the unlock alive.
    if(aud) return;
    aud=new Audio();
    aud.preload='auto';
    aud.addEventListener('ended', function(){{
      var i=cur;
      dot(i,'done');
      tmr=setTimeout(function(){{ playIdx(i+1); }}, pauseMs);
    }});
    aud.addEventListener('error', function(){{
      tmr=setTimeout(function(){{ playIdx(cur+1); }}, 300);
    }});
  }}
  function stop(){{
    if(aud){{ try{{aud.pause();}}catch(e){{}} }}
    if(tmr){{clearTimeout(tmr); tmr=null;}}
    playing=false; cur=-1;
    document.getElementById('pl-btn-'+uid).textContent='▶ Play All';
    document.getElementById('pl-btn-'+uid).style.color='#4F46E5';
  }}
  function playIdx(i){{
    if(i>=n){{
      stop();
      document.getElementById('pl-stat-'+uid).textContent='done ✓';
      for(let j=0;j<n;j++) dot(j,'done');
      return;
    }}
    cur=i; playing=true;
    for(let j=0;j<i;j++) dot(j,'done'); dot(i,'active');
    document.getElementById('pl-stat-'+uid).textContent='▶ '+(i+1)+' / '+n;
    if(!srcs[i]){{ tmr=setTimeout(function(){{ playIdx(i+1); }}, pauseMs); return; }}
    aud.src=srcs[i];
    var p=aud.play();
    if(p && typeof p.catch === 'function'){{
      p.catch(function(){{ tmr=setTimeout(function(){{ playIdx(i+1); }}, 300); }});
    }}
  }}
  window['plToggle_'+uid]=function(){{
    if(playing){{ stop(); document.getElementById('pl-stat-'+uid).textContent='stopped'; }}
    else{{
      ensureAud();  // must run during this user-gesture click
      document.getElementById('pl-btn-'+uid).textContent='■ Stop';
      document.getElementById('pl-btn-'+uid).style.color='#FF7B6B';
      playIdx(0);
    }}
  }};
}})();
</script>
"""


def lessons_table(rows, active_idx=None, scores=None,
                  show_word=True, show_trans=True):
    """Compact table view of all rows in a lesson (used by steps 1, 2, 4, 5)."""
    html_rows = ""
    for i, (_, r) in enumerate(rows.iterrows()):
        word  = r["word"] if show_word else "—"
        trans = r["transcription"] if show_trans else "—"
        score_html = ""
        if scores and i in scores:
            s = scores[i]
            color = "var(--mova-mint)" if s.get("passed") else "var(--mova-coral-ink)"
            pct   = int(s.get("score", 0) * 100)
            score_html = (f'<span style="background:{"var(--mova-mint-soft)" if s.get("passed") else "var(--mova-coral-soft)"};'
                          f'color:{color};border-radius:5px;padding:2px 9px;'
                          f'font-family:JetBrains Mono,monospace;font-size:.78rem">{pct}%</span>')
        style = ""
        if active_idx == i:
            style = "background:var(--mova-indigo-soft);border-left:3px solid var(--mova-indigo);"
        html_rows += (
            f'<div class="row-ok" style="{style}">'
            f'<span style="min-width:28px;color:var(--mova-indigo-ink);font-family:JetBrains Mono,monospace;font-size:.75rem">{i+1:02d}</span>'
            f'<span style="flex:1;color:var(--mova-ink);font-size:1.05rem">{word}</span>'
            f'<span style="flex:1;color:var(--mova-indigo);font-family:JetBrains Mono,monospace;font-size:.9rem">{trans}</span>'
            f'{score_html}'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:var(--mova-card);border:1px solid var(--mova-line-2);border-radius:12px;'
        f'overflow:hidden;margin:8px 0">{html_rows}</div>',
        unsafe_allow_html=True,
    )


def preload_lesson_audio(rows, prefix: str):
    """Cache audio paths in session_state under `{prefix}_paths` (list of str or None)."""
    lang = _r_lang()
    key  = f"{prefix}_paths"
    if key not in st.session_state:
        with st.spinner("Готуємо аудіо..."):
            paths = []
            for _, r in rows.iterrows():
                p = audio_for_row(r["word"], r["transcription"],
                                  lesson_id=int(r["lesson_id"]), lang=lang)
                paths.append(str(p) if p else None)
        st.session_state[key] = paths
    return [Path(p) if p else None for p in st.session_state[key]]


def mic(uid: str):
    if hasattr(st, "audio_input"):
        r = st.audio_input("🎙️", key=f"mic_{uid}")
        return r.read() if r else None
    f = st.file_uploader("Upload audio", type=["webm", "wav", "mp3"],
                         key=f"up_{uid}", label_visibility="collapsed")
    return f.read() if f else None


@st.cache_data
def load(path: str, lang: str = "en", native_lang: str = "Ukrainian") -> pd.DataFrame:
    """Load lesson data for the given target language.

    For English: reads the 'en' sheet (5 cols) and merges rules from
    the 'Правила' sheet in the user's native language.
    For uk/es/ko: reads the respective sheet (4 cols, no IPA transcription for ko).
    """
    df = pd.read_excel(path, engine="openpyxl", sheet_name=lang)

    if lang == "en":
        df = df.iloc[:, :5]
        df.columns = ["lesson_id", "row_id", "word", "transcription", "rule"]
        # Merge multilingual rules from «Правила» sheet
        rules_col = NATIVE_TO_RULES_COL.get(native_lang, "en")
        try:
            df_rules = pd.read_excel(path, engine="openpyxl", sheet_name="Правила")
            df_rules.columns = ["lesson_id", "ru", "en", "es", "uk"]
            rule_map = dict(zip(df_rules["lesson_id"].astype(int),
                                df_rules[rules_col].fillna("")))
            # Apply: Правила sheet takes priority (multilingual); fall back to inline rule
            def _apply_rule(row):
                from_sheet = rule_map.get(int(row["lesson_id"]), "")
                if from_sheet:
                    return from_sheet
                existing = str(row["rule"]).strip() if pd.notna(row["rule"]) else ""
                if existing and existing.lower() != "nan":
                    return existing
                return ""
            df["rule"] = df.apply(_apply_rule, axis=1)
        except Exception as e:
            print(f"[load] rules merge failed: {e}")
            df["rule"] = df["rule"].fillna("").astype(str).str.strip()
    elif lang == "ko":
        df = df.iloc[:, :3]
        df.columns = ["lesson_id", "row_id", "word"]
        df["transcription"] = ""
        df["rule"]          = ""
    else:  # uk, es
        df = df.iloc[:, :4]
        df.columns = ["lesson_id", "row_id", "word", "transcription"]
        df["rule"] = ""

    df["lesson_id"]     = pd.to_numeric(df["lesson_id"], errors="coerce").fillna(0).astype(int)
    df["word"]          = df["word"].astype(str).str.strip()
    df["transcription"] = df["transcription"].astype(str).str.strip().replace("nan", "")
    df["rule"]          = df["rule"].fillna("").astype(str).str.strip().replace("nan", "")
    df = df[df["lesson_id"] > 0].reset_index(drop=True)
    return df


# st.set_page_config is set up by main_app.py when used as a launcher.
# When this file is run directly, set it here too.
try:
    st.set_page_config(page_title="Reading Practice", page_icon="📖",
                       layout="wide", initial_sidebar_state="collapsed")
except Exception:
    pass  # Already set by main_app.py

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
/* removed: was fighting Mova surface; theme is now driven by tokens.css */
#MainMenu,footer{visibility:hidden;}
/* Keep Streamlit's sidebar collapse/expand control reachable on every device,
   including iOS Safari, where the control would otherwise be invisible. */
header{background:transparent !important;}
header [data-testid="stDecoration"]{display:none;}
/* Floating sidebar toggle (visible whenever the sidebar is collapsed) */
[data-testid="collapsedControl"]{
    visibility:visible !important;
    opacity:1 !important;
    display:flex !important;
    z-index:9999 !important;
    position:fixed !important;
    top:0.6rem !important;
    left:0.6rem !important;
    background:var(--mova-card) !important;
    border:1px solid var(--mova-indigo) !important;
    border-radius:8px !important;
    box-shadow:0 2px 8px rgba(0,0,0,.4) !important;
}
[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] svg{
    color:var(--mova-indigo) !important;
    fill:var(--mova-indigo) !important;
    min-width:36px !important;
    min-height:36px !important;
}
.wcard{background:var(--mova-card);border:1px solid var(--mova-line);border-radius:14px;padding:28px 20px;margin:10px 0;text-align:center;}
.wbig{font-size:3.2rem;font-weight:700;color:var(--mova-ink);}
.tbig{font-size:2rem;color:var(--mova-indigo);font-family:'JetBrains Mono',monospace;margin-top:8px;}
.rule{background:var(--mova-card);border-left:3px solid var(--mova-indigo);border-radius:6px;padding:10px 14px;margin:8px 0;color:#a0a0d0;font-size:.9rem;}
.spill{font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:3px 10px;border-radius:20px;margin:2px;display:inline-block;}
.row-ok{display:flex;gap:10px;padding:8px 14px;background:var(--mova-card);border-bottom:1px solid var(--mova-line);align-items:center;}
/* Make Streamlit secondary buttons (e.g. step 3 choices) dark-themed for readability */
.stApp .stButton > button[kind="secondary"]{
    background:var(--mova-card) !important;
    color:var(--mova-ink) !important;
    border:1px solid var(--mova-line) !important;
    font-weight:500 !important;
}
.stApp .stButton > button[kind="secondary"]:hover{
    background:var(--mova-indigo-soft) !important;
    border-color:var(--mova-indigo) !important;
    color:var(--mova-ink) !important;
}
</style>
""", unsafe_allow_html=True)


STEPS = {
    1: "Послухай і повтори",
    2: "Прочитай слова",
    3: "Послухай і знайди",
    4: "Послухай і повтори",
    5: "Прочитай на час",
}
REQUIRED = {1, 2, 3}


def current_step() -> int:
    return st.session_state.get("r_step", 1)


def shdr(step: int):
    pills = ""
    for s in range(1, 6):
        if s == current_step():
            style = "background:var(--mova-indigo-soft);color:var(--mova-indigo);border:1px solid var(--mova-indigo)"
        elif s < current_step():
            style = "background:var(--mova-mint-soft);color:var(--mova-mint);border:1px solid var(--mova-mint)"
        elif s in REQUIRED:
            style = "background:var(--mova-amber-soft);color:var(--mova-amber-ink);border:1px solid var(--mova-amber)"
        else:
            style = "background:var(--mova-card);color:var(--mova-ink-3);border:1px solid var(--mova-line)"
        lbl = f"{'🔒' if s in REQUIRED and s > current_step() else s}"
        pills += f'<span class="spill" style="{style}">{lbl}</span>'

    req_note = ""
    if step in REQUIRED:
        req_note = ' <span style="color:var(--mova-amber-ink);font-size:.72rem">🔒 обов\'язковий</span>'

    st.markdown(f'<div style="margin-bottom:10px">{pills}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:var(--mova-card);'
        f'border:1px solid var(--mova-line);border-radius:14px;padding:14px 20px;margin-bottom:14px">'
        f'<div style="color:var(--mova-indigo);font-size:.75rem;font-family:JetBrains Mono,monospace">'
        f'КРОК {step} / 5{req_note}</div>'
        f'<div style="color:var(--mova-ink);font-size:1.15rem;font-weight:600">{STEPS[step]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def card(word, trans, rule="", show_word=True, show_trans=True):
    w = f'<div class="wbig">{word}</div>' if show_word else ""
    t = f'<div class="tbig">{trans}</div>' if show_trans else ""
    r = f'<div class="rule">📖 {rule}</div>' if rule else ""
    st.markdown(f'<div class="wcard">{w}{t}{r}</div>', unsafe_allow_html=True)


def pbar(val: float):
    val = max(0.0, min(1.0, val))
    st.markdown(
        f'<div style="background:var(--mova-card);border-radius:6px;height:6px;overflow:hidden;margin:6px 0">'
        f'<div style="height:6px;background:linear-gradient(90deg, var(--mova-indigo), #6E66FF);width:{val*100:.0f}%"></div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1 — Послухай і повтори (all on screen, autoplay with 1s pause)
# ═══════════════════════════════════════════════════════════════════════════

def do_step1(rows: pd.DataFrame) -> bool:
    shdr(1)

    # Show rule if present (any row has one) — use first non-empty
    rule_txt = next((r["rule"] for _, r in rows.iterrows() if r["rule"]), "")
    if rule_txt:
        st.markdown(f'<div class="rule">📖 {rule_txt}</div>', unsafe_allow_html=True)

    # Continue button ABOVE the player — user can skip ahead without scrolling.
    if st.button("Продовжити →", type="primary", use_container_width=True,
                 key="s1_done"):
        return True

    # Combined player + word-list with active-word highlight (reused from grammar).
    from grammar import autoplaylist_with_table
    paths = preload_lesson_audio(rows, "s1")
    phrase_dicts = [
        {"native": str(r["word"]), "target": str(r["transcription"])}
        for _, r in rows.iterrows()
    ]
    pauses = [1.0 + 0.2 * max(0, len(str(r["word"])) - 2)
              for _, r in rows.iterrows()]
    height = 200 + 48 * len(rows)
    components.html(
        autoplaylist_with_table(phrase_dicts, paths, pauses, uid="rs1",
                                show_native=True, show_target=True),
        height=height, scrolling=True,
    )
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2 — Прочитай слова (all on screen, mic + similarity check)
# ═══════════════════════════════════════════════════════════════════════════

def do_step2(rows: pd.DataFrame) -> bool:
    shdr(2)
    st.markdown(
        '<div style="color:var(--mova-ink-2);font-size:.9rem;margin:-6px 0 14px">'
        'Запиши себе вголос і перевір вимову — або просто прочитай очима і натисни Далі.</div>',
        unsafe_allow_html=True,
    )

    scores = st.session_state.get("s2_scores", {})

    rule_txt = next((r["rule"] for _, r in rows.iterrows() if r["rule"]), "")
    if rule_txt:
        st.markdown(f'<div class="rule">📖 {rule_txt}</div>', unsafe_allow_html=True)

    # ── Mic ABOVE phrases ────────────────────────────────────────────────────
    expected = ". ".join(str(r["word"]).strip() for _, r in rows.iterrows())
    audio = mic("s2")

    if not STT_OK or not SCORER_OK:
        st.caption("⚠️ Для перевірки потрібно: `pip install openai-whisper rapidfuzz`")

    if st.button("✓ Перевірити вимову", type="primary",
                 use_container_width=True, key="s2_check"):
        if not audio:
            st.warning("Спочатку запиши аудіо!")
        elif not STT_OK or not SCORER_OK:
            st.warning("Whisper/RapidFuzz не встановлені.")
        else:
            t_ms = _audio_duration_ms(audio)
            with st.spinner("Розпізнаємо мовлення..."):
                r = score_audio(audio, expected)
            if r:
                scores = {i: r for i in range(len(rows))}
                st.session_state["s2_scores"] = scores
                color = "var(--mova-mint)" if r["passed"] else "var(--mova-coral-ink)"
                st.markdown(
                    f'<div style="text-align:center;font-size:1.6rem;'
                    f'color:{color};font-weight:600">{int(r["score"]*100)}%</div>',
                    unsafe_allow_html=True,
                )
                _log_score(step=2, phrase_id=0,
                           similarity=r["score"],
                           response_time_ms=t_ms,
                           success=bool(r["passed"]))
                st.rerun()
            else:
                st.error("Не вдалося розпізнати аудіо.")

    # ── Phrases table BELOW mic ───────────────────────────────────────────────
    lessons_table(rows, show_word=True, show_trans=True, scores=scores)

    # ── Далі at the very bottom ───────────────────────────────────────────────
    if st.button("Далі →", use_container_width=True, key="s2_next"):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3 — Послухай і знайди
# ═══════════════════════════════════════════════════════════════════════════

def do_step3(rows: pd.DataFrame) -> bool:
    shdr(3)

    if "s3_init" not in st.session_state:
        st.session_state["s3_init"]    = True
        st.session_state["s3_idx"]     = 0
        st.session_state["s3_scores"]  = {}
        shuffled = {}
        for i in range(len(rows)):
            opts = list(rows["word"].values)
            random.shuffle(opts)
            shuffled[i] = opts
        st.session_state["s3_shuffled"] = shuffled
        for i, (_, row) in enumerate(rows.iterrows()):
            akey = f"s3_audio_{i}"
            if akey not in st.session_state:
                p = audio_for_row(row["word"], row["transcription"],
                                  lesson_id=int(row["lesson_id"]))
                st.session_state[akey] = str(p) if p else None

    idx      = st.session_state["s3_idx"]
    scores   = st.session_state["s3_scores"]
    shuffled = st.session_state["s3_shuffled"]

    if scores:
        html = "".join(
            f'<div class="row-ok">'
            f'<span style="color:{"var(--mova-mint)" if v else "var(--mova-coral-ink)"};flex:1">{"✓" if v else "✗"} {rows.iloc[i]["word"]}</span>'
            f'<span style="color:var(--mova-ink-3);font-family:JetBrains Mono,monospace;font-size:.78rem">{rows.iloc[i]["transcription"]}</span>'
            f'</div>'
            for i, v in sorted(scores.items())
        )
        st.markdown(
            f'<div style="border-radius:10px;overflow:hidden;margin:8px 0">{html}</div>',
            unsafe_allow_html=True,
        )

    if idx < len(rows):
        row = rows.iloc[idx]
        p   = st.session_state.get(f"s3_audio_{idx}")

        st.markdown(f"**Послухай слово {idx+1} і знайди його:**")
        if p:
            play(p, autoplay=True)
        else:
            st.caption("⚠️ Аудіо недоступне")

        c_rp, _ = st.columns([1, 3])
        with c_rp:
            if p and st.button("▶ Ще раз", key=f"s3_rp_{idx}"):
                play(p)

        st.markdown("---")
        opts = shuffled[idx]
        cols = st.columns(2)
        for ci, choice in enumerate(opts):
            with cols[ci % 2]:
                if st.button(choice, key=f"s3_ch_{idx}_{ci}",
                             use_container_width=True):
                    ok = choice.strip().lower() == row["word"].strip().lower()
                    scores[idx] = ok
                    st.session_state["s3_scores"] = scores
                    st.session_state["s3_idx"]    = idx + 1
                    _log_score(step=3, phrase_id=int(row.get("row_id", idx + 1)),
                               similarity=1.0 if ok else 0.0,
                               response_time_ms=0,
                               success=ok)
                    if ok:
                        st.success(f"✓ Правильно! — {row['transcription']}")
                    else:
                        st.error(f"✗ Неправильно. Правильно: **{row['word']}** {row['transcription']}")
                    time.sleep(0.4)
                    st.rerun()
        return False

    ok = sum(1 for v in scores.values() if v)
    st.success(f"✓ Готово! {ok}/{len(rows)}")
    if st.button("Продовжити →", type="primary", use_container_width=True, key="s3_done"):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4 — Послухай і повтори (all on screen, autoplay with 1s pause)
# ═══════════════════════════════════════════════════════════════════════════

def do_step4(rows: pd.DataFrame) -> bool:
    shdr(4)

    rule_txt = next((r["rule"] for _, r in rows.iterrows() if r["rule"]), "")
    if rule_txt:
        st.markdown(f'<div class="rule">📖 {rule_txt}</div>', unsafe_allow_html=True)

    # Continue / Skip buttons ABOVE the player
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Продовжити →", type="primary",
                     use_container_width=True, key="s4_done"):
            return True
    with c2:
        if st.button("⏭ Пропустити", key="s4_skip", use_container_width=True):
            return True

    # Combined player + word-list with active-word highlight (reused from grammar).
    from grammar import autoplaylist_with_table
    paths = preload_lesson_audio(rows, "s4")
    phrase_dicts = [
        {"native": str(r["word"]), "target": str(r["transcription"])}
        for _, r in rows.iterrows()
    ]
    pauses = [1.0 + 0.2 * max(0, len(str(r["word"])) - 2)
              for _, r in rows.iterrows()]
    height = 200 + 48 * len(rows)
    components.html(
        autoplaylist_with_table(phrase_dicts, paths, pauses, uid="rs4",
                                show_native=True, show_target=True),
        height=height, scrolling=True,
    )
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5 — Прочитай на час (mic-driven timer + similarity check)
# ═══════════════════════════════════════════════════════════════════════════

def do_step5(rows: pd.DataFrame) -> bool:
    shdr(5)

    # Mic FIRST so mobile users don't need to scroll past the word grid
    st.markdown("#### 🎙️ Запиши себе, поки читаєш вголос всі слова")
    audio = mic("s5")

    # Show all words as grid
    chips = "".join(
        f'<span style="font-size:1.3rem;font-weight:600;color:var(--mova-ink);'
        f'background:var(--mova-card);border:1px solid var(--mova-line);border-radius:10px;'
        f'padding:10px 16px;margin:4px;display:inline-block">'
        f'{row["word"]}'
        f'<span style="display:block;font-size:.75rem;color:var(--mova-ink-3);'
        f'font-family:JetBrains Mono,monospace">{row["transcription"]}</span></span>'
        for _, row in rows.iterrows()
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;padding:16px;'
        f'background:var(--mova-surface);border-radius:12px">{chips}</div>',
        unsafe_allow_html=True,
    )

    if not STT_OK or not SCORER_OK:
        st.caption("⚠️ Для перевірки вимови потрібно: `pip install openai-whisper rapidfuzz`")

    expected = ". ".join(str(r["word"]).strip() for _, r in rows.iterrows())

    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("✓ Завершити та перевірити", type="primary",
                     key="s5_sub", use_container_width=True):
            if not audio:
                st.warning("Спочатку запиши аудіо!")
            else:
                t_ms = _audio_duration_ms(audio)
                res = {"time": max(1, round(t_ms / 1000))}
                if STT_OK and SCORER_OK:
                    with st.spinner("Перевіряємо вимову..."):
                        r = score_audio(audio, expected)
                    if r:
                        res["score"]  = r["score"]
                        res["passed"] = r["passed"]
                # Log step 5 outcome
                _log_score(
                    step=5, phrase_id=0,
                    similarity=res.get("score", 0.0),
                    response_time_ms=t_ms,
                    success=bool(res.get("passed", False)),
                )
                st.session_state["s5_result"] = res
                st.rerun()
    with c2:
        if st.button("Пропустити", key="s5_skip", use_container_width=True):
            return True

    if "s5_result" in st.session_state:
        res = st.session_state["s5_result"]
        score_str = f" · {int(res['score']*100)}% точність" if "score" in res else ""
        emoji = "🎉" if res.get("passed") else "🏁"
        st.success(f"{emoji} {res['time']} секунд{score_str}")
        if st.button("Завершити урок ✓", type="primary",
                     use_container_width=True, key="s5_fin"):
            st.session_state.pop("s5_result", None)
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  State management
# ═══════════════════════════════════════════════════════════════════════════

STEP_FNS = {1: do_step1, 2: do_step2, 3: do_step3, 4: do_step4, 5: do_step5}


def clear_step_state():
    """Remove all per-step keys but keep lesson/user/rows."""
    keep = {"r_step", "r_lesson", "r_user", "r_rows"}
    for k in list(st.session_state):
        if k not in keep and (
            k.startswith("s1_") or k.startswith("s2_") or
            k.startswith("s3_") or k.startswith("s4_") or
            k.startswith("s5_") or k.startswith("mic_") or k.startswith("up_")
        ):
            del st.session_state[k]


def clear_all():
    for k in list(st.session_state):
        del st.session_state[k]
    st.query_params.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  Setup screen
# ═══════════════════════════════════════════════════════════════════════════

def _render_module_nav_sidebar(current_module: str) -> None:
    """Render the module-switcher sidebar (shared by setup and active-lesson views)."""
    _MODS = [
        ("grammar", "🗣️", "Grammar"),
        ("vocab",   "📖", "Vocabulary"),
        ("reading", "🔤", "Reading"),
        ("custom",  "📝", "My Phrases"),
    ]
    st.markdown(
        '<div style="font-size:.7rem;color:var(--mova-ink-3);'
        'text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">'
        'Module</div>',
        unsafe_allow_html=True,
    )
    for _mk, _mi, _mn in _MODS:
        _active = (_mk == current_module)
        if st.button(
            f"{_mi} {_mn}",
            key=f"sb_nav_{_mk}",
            use_container_width=True,
            type="primary" if _active else "secondary",
            disabled=_active,
        ):
            _u = st.session_state.get("launcher_user", "student1")
            _n = st.session_state.get("launcher_native", "Ukrainian")
            _t = st.session_state.get("launcher_target", "English")
            for _k in list(st.session_state):
                del st.session_state[_k]
            st.session_state.update({
                "active_module":   _mk,
                "launcher_user":   _u,
                "launcher_native": _n,
                "launcher_target": _t,
            })
            st.query_params["module"] = _mk
            st.rerun()
    st.markdown("---")


def render_setup():
    with st.sidebar:
        _render_module_nav_sidebar("reading")
        if st.button("🏠 Main menu", key="r_setup_home"):
            for _k in list(st.session_state):
                del st.session_state[_k]
            st.query_params.clear()
            st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:36px 0 20px">
      <div style="font-size:3rem">📖</div>
      <h1 style="color:var(--mova-ink);font-weight:600;margin:10px 0 4px">Reading Practice</h1>
      <p style="color:var(--mova-ink-3)">Фонетика · IPA озвучка · 4 мови</p>
    </div>""", unsafe_allow_html=True)

    if not _edge_ok() and not _gtts_ok():
        st.error("⚠️ Встанови аудіо бібліотеку:\n\n`pip install edge-tts`\n\nабо\n\n`pip install gtts`")

    # ── Language selector ──────────────────────────────────────────────────
    native_lang = st.session_state.get("launcher_native", "Ukrainian")
    lang_options = list(LANG_LABELS.keys())
    saved_lang   = st.session_state.get("r_lang", "en")
    lang_idx     = lang_options.index(saved_lang) if saved_lang in lang_options else 0

    col_lang, col_user = st.columns([2, 1])
    with col_lang:
        chosen_lang = st.selectbox(
            "🌐 Мова для вивчення",
            lang_options,
            index=lang_idx,
            format_func=lambda k: LANG_LABELS[k],
            key="r_lang_select",
        )
    with col_user:
        default_user = st.session_state.get("launcher_user", "student1")
        user_id = st.text_input("👤 Ім'я", value=default_user)

    # Reload data when language changes
    if chosen_lang != st.session_state.get("r_lang"):
        st.session_state["r_lang"] = chosen_lang
        st.rerun()

    # Load data for selected language
    df = load(str(DB_PATH), lang=chosen_lang, native_lang=native_lang)
    lessons = sorted(df["lesson_id"].unique())

    # Auto-select lesson based on saved progress
    progress    = None
    default_idx = 0
    resume_step = 1
    resume_msg  = None
    if LOGGER_OK and user_id:
        try:
            progress = get_progress(user_id, _reading_lang_pair())
        except Exception:
            progress = None

    if progress:
        saved_lesson = progress["last_completed_lesson"]
        saved_step   = progress["last_step"]
        if saved_step >= 99:
            next_lesson = saved_lesson + 1
            if next_lesson in lessons:
                default_idx = lessons.index(next_lesson)
                resume_step = 1
                resume_msg  = f"▶ Продовжуєш з уроку {next_lesson} (останній пройдений: {saved_lesson})"
        else:
            if saved_lesson in lessons:
                default_idx = lessons.index(saved_lesson)
                resume_step = max(1, min(5, saved_step))
                resume_msg  = f"⏯ Повернешся до уроку {saved_lesson} на крок {resume_step}"

    lesson_id = st.selectbox(
        "📚 Урок", lessons,
        index=default_idx,
        format_func=lambda x: f"Урок {x} — {len(df[df['lesson_id']==x])} рядків",
    )

    rows = df[df["lesson_id"] == lesson_id].reset_index(drop=True)
    has_trans = chosen_lang != "ko"  # Korean has no transcription

    st.markdown(f"**{len(rows)} слів/рядків у цьому уроці:**")
    preview = "".join(
        '<div style="display:flex;gap:14px;padding:8px 14px;background:var(--mova-card);'
        'border-bottom:1px solid var(--mova-line);align-items:center">'
        f'<span style="min-width:24px;color:var(--mova-indigo-ink);font-family:JetBrains Mono,monospace;font-size:.75rem">{i+1:02d}</span>'
        f'<span style="flex:1;font-size:1rem;color:var(--mova-ink)">{row["word"]}</span>'
        + (f'<span style="color:var(--mova-indigo);font-family:JetBrains Mono,monospace;font-size:.85rem">{row["transcription"]}</span>'
           if has_trans and row["transcription"] else '')
        + ('<span style="color:var(--mova-ink-3);font-size:.75rem;margin-left:8px">'
           + row["rule"][:40] + '...</span>' if len(row["rule"]) > 5 else '')
        + '</div>'
        for i, (_, row) in enumerate(rows.iterrows())
    )
    st.markdown(
        f'<div style="border-radius:10px;overflow:hidden;max-height:280px;overflow-y:auto">'
        f'{preview}</div>',
        unsafe_allow_html=True,
    )

    start_at_step = resume_step if (progress and lesson_id == lessons[default_idx]) else 1
    if resume_msg and lesson_id == lessons[default_idx]:
        st.info(resume_msg)

    st.markdown("")
    btn_label = f"▶ Продовжити з кроку {start_at_step}" if start_at_step > 1 else "▶ Почати урок"
    if st.button(btn_label, type="primary", use_container_width=True):
        st.session_state["r_lang"]   = chosen_lang
        st.session_state["r_lesson"] = int(lesson_id)
        st.session_state["r_user"]   = user_id
        st.session_state["r_rows"]   = rows
        st.session_state["r_step"]   = start_at_step
        st.session_state.pop("_r_progress_saved", None)
        st.session_state.pop("_r_last_saved_progress", None)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
/* removed: was fighting Mova surface; theme is now driven by tokens.css */
#MainMenu,footer{visibility:hidden;}
/* Keep Streamlit's sidebar collapse/expand control reachable on every device,
   including iOS Safari, where the control would otherwise be invisible. */
header{background:transparent !important;}
header [data-testid="stDecoration"]{display:none;}
/* Floating sidebar toggle (visible whenever the sidebar is collapsed) */
[data-testid="collapsedControl"]{
    visibility:visible !important;
    opacity:1 !important;
    display:flex !important;
    z-index:9999 !important;
    position:fixed !important;
    top:0.6rem !important;
    left:0.6rem !important;
    background:var(--mova-card) !important;
    border:1px solid var(--mova-indigo) !important;
    border-radius:8px !important;
    box-shadow:0 2px 8px rgba(0,0,0,.4) !important;
}
[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] svg{
    color:var(--mova-indigo) !important;
    fill:var(--mova-indigo) !important;
    min-width:36px !important;
    min-height:36px !important;
}
.wcard{background:var(--mova-card);border:1px solid var(--mova-line);border-radius:14px;padding:28px 20px;margin:10px 0;text-align:center;}
.wbig{font-size:3.2rem;font-weight:700;color:var(--mova-ink);}
.tbig{font-size:2rem;color:var(--mova-indigo);font-family:'JetBrains Mono',monospace;margin-top:8px;}
.rule{background:var(--mova-card);border-left:3px solid var(--mova-indigo);border-radius:6px;padding:10px 14px;margin:8px 0;color:#a0a0d0;font-size:.9rem;}
.spill{font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:3px 10px;border-radius:20px;margin:2px;display:inline-block;}
.row-ok{display:flex;gap:10px;padding:8px 14px;background:var(--mova-card);border-bottom:1px solid var(--mova-line);align-items:center;}
/* Make Streamlit secondary buttons (e.g. step 3 choices) dark-themed for readability */
.stApp .stButton > button[kind="secondary"]{
    background:var(--mova-card) !important;
    color:var(--mova-ink) !important;
    border:1px solid var(--mova-line) !important;
    font-weight:500 !important;
}
.stApp .stButton > button[kind="secondary"]:hover{
    background:var(--mova-indigo-soft) !important;
    border-color:var(--mova-indigo) !important;
    color:var(--mova-ink) !important;
}
</style>
""", unsafe_allow_html=True)


def main():
    _inject_css()
    if not DB_PATH.exists():
        st.error(
            f"**Файл не знайдено:** `{DB_PATH}`\n\n"
            "Скопіюй Excel файл у `data/reading_lessons.xlsx`."
        )
        st.stop()

    if "r_step" not in st.session_state:
        render_setup()
        return

    lang        = _r_lang()
    native_lang = st.session_state.get("launcher_native", "Ukrainian")
    df          = load(str(DB_PATH), lang=lang, native_lang=native_lang)

    step = st.session_state["r_step"]
    rows = st.session_state["r_rows"]

    # Auto-save progress on every step (for resume next session)
    cur_lesson = st.session_state.get("r_lesson", 0)
    cur_user   = st.session_state.get("r_user", "anonymous")
    if step > 5:
        _save_step_progress(cur_lesson, 99, cur_user)  # 99 = lesson done
    else:
        _save_step_progress(cur_lesson, step, cur_user)

    with st.sidebar:
        _render_module_nav_sidebar("reading")
        st.markdown("**🔤 Reading**")
        all_l    = sorted(df["lesson_id"].unique())
        lid      = st.session_state.get("r_lesson", 1)
        total    = max(len(all_l), 1)
        # Lessons fully completed = lid - 1 (current one is in progress)
        completed = max(lid - 1, 0)
        pct       = round(completed / total * 100, 1)

        st.markdown(
            f'<div style="background:var(--mova-card);border:1px solid var(--mova-line);'
            f'border-radius:10px;padding:10px 12px;margin:4px 0 10px">'
            f'<div style="color:var(--mova-ink-2);font-size:.7rem;'
            f'font-family:\'JetBrains Mono\',monospace;'
            f'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">'
            f'Твій шлях · Reading</div>'
            f'<div style="color:var(--mova-ink);font-size:1.05rem;font-weight:600">'
            f'Урок {lid} / {total}</div>'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:.72rem;'
            f'color:var(--mova-ink-3);margin:6px 0 2px">'
            f'<span>{completed} пройдено · {total - completed} попереду</span>'
            f'<span>{pct}%</span></div>'
            f'<div style="background:var(--mova-card);border-radius:6px;height:6px;overflow:hidden">'
            f'<div style="height:6px;background:linear-gradient(90deg, var(--mova-indigo), #6E66FF);'
            f'width:{pct}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Step indicator
        step_pct = round((step - 1) / 5 * 100, 0)
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:.72rem;'
            f'color:var(--mova-ink-3);margin-bottom:2px">'
            f'<span>Крок {step} / 5 — {STEPS.get(step,"")}</span>'
            f'<span>{"🔒" if step in REQUIRED else ""}</span></div>'
            f'<div style="background:var(--mova-card);border-radius:6px;height:6px;overflow:hidden">'
            f'<div style="height:6px;background:linear-gradient(90deg, var(--mova-mint), #34D0A0);'
            f'width:{step_pct}%"></div></div>',
            unsafe_allow_html=True,
        )

        # ── Step navigation: Previous / Repeat / Jump ──
        st.markdown("---")
        st.caption("Навігація між кроками")
        nav_c1, nav_c2 = st.columns(2)
        with nav_c1:
            back_disabled = step <= 1
            if st.button("← Попередній", disabled=back_disabled,
                         use_container_width=True, key="r_nav_back",
                         help="Повернутися до попереднього кроку"):
                clear_step_state()
                st.session_state["r_step"] = max(1, step - 1)
                st.rerun()
        with nav_c2:
            if st.button("🔄 Повторити", use_container_width=True,
                         key="r_nav_repeat",
                         help="Перезапустити поточний крок"):
                clear_step_state()
                st.rerun()

        jump_default = min(max(step, 1), 5) - 1
        jump_to = st.selectbox(
            "Перейти до кроку",
            options=list(range(1, 6)),
            index=jump_default,
            format_func=lambda s: f"Крок {s}" + (" 🔒" if s in REQUIRED else ""),
            key="r_nav_jump",
        )
        if jump_to != step:
            if st.button(f"Перейти до кроку {jump_to}",
                         use_container_width=True, key="r_nav_go"):
                clear_step_state()
                st.session_state["r_step"] = jump_to
                st.rerun()

        # ── Jump to lesson ────────────────────────────────────────────────────
        st.caption("Перейти до уроку")
        _jump_lid = st.selectbox(
            "lesson_jump_sel_r",
            options=all_l,
            index=all_l.index(lid) if lid in all_l else 0,
            format_func=lambda l: f"Урок {l}",
            key="sb_r_jump_lid",
            label_visibility="collapsed",
        )
        if _jump_lid != lid:
            if st.button(
                f"Перейти до уроку {_jump_lid}",
                use_container_width=True,
                key="sb_r_go_lid",
            ):
                clear_step_state()
                st.session_state["r_lesson"] = _jump_lid
                st.session_state["r_step"]   = 1
                st.rerun()

        st.markdown("---")
        if st.button("🏠 Головне меню"):
            clear_all()
            st.rerun()

    if step > 5:
        # Save progress (idempotent guard)
        if LOGGER_OK and not st.session_state.get("_r_progress_saved"):
            try:
                save_progress(
                    user_id              = st.session_state.get("r_user", "anonymous"),
                    language_pair        = _reading_lang_pair(),
                    last_completed_lesson= int(st.session_state.get("r_lesson", 0)),
                    last_step            = 99,
                )
                st.session_state["_r_progress_saved"] = True
            except Exception as e:
                print(f"[reading_app] save_progress failed: {e}")

        st.markdown("""
        <div style="background:linear-gradient(135deg, var(--mova-mint-soft), var(--mova-indigo-soft));
             border:1px solid var(--mova-mint);border-radius:16px;padding:40px;text-align:center">
          <div style="font-size:3rem">&#x1F389;</div>
          <h2 style="color:var(--mova-ink)">Урок завершено!</h2>
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("\U0001f504 Повторити", type="primary", use_container_width=True):
                clear_step_state()
                st.session_state["r_step"] = 1
                st.session_state.pop("_r_progress_saved", None)
                st.session_state.pop("_r_last_saved_progress", None)
                st.rerun()
        with c2:
            if st.button("📚 Новий урок", use_container_width=True):
                clear_all()
                st.rerun()
        return

    fn   = STEP_FNS.get(step)
    done = fn(rows) if fn else True

    if done:
        clear_step_state()
        st.session_state["r_step"] = step + 1
        st.rerun()
    elif step not in REQUIRED and step != 4:
        # Step 4 has its own "Пропустити" button inside do_step4; don't duplicate it.
        st.markdown("---")
        if st.button(f"⏭ Пропустити крок {step}", key=f"skip_global_{step}"):
            clear_step_state()
            st.session_state["r_step"] = step + 1
            st.rerun()


if __name__ == "__main__":
    main()
