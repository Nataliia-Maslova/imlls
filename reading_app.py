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
PHONEMES_DIR = ROOT / "phonemes" / "en"

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


def score_audio(audio_bytes, expected_text):
    """Transcribe via Whisper and score similarity vs expected_text."""
    if not STT_OK or not SCORER_OK or not audio_bytes:
        return None
    try:
        text = transcribe_bytes(audio_bytes, language="en")
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


def _gtts(text: str, path: Path):
    from gtts import gTTS
    gTTS(text=text, lang="en", slow=True).save(str(path))


async def _edge(text: str, path: Path, rate: str = "-5%"):
    import edge_tts
    tts = edge_tts.Communicate(text, voice="en-US-JennyNeural", rate=rate)
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


def audio_for_word(word: str):
    """Generate MP3 for a word or compound phrase. Cached permanently."""
    # Strip stress markers / curly apostrophes that TTS doesn't handle well
    clean = word
    for ch in ("'", "‘", "’", "`"):
        clean = clean.replace(ch, "")
    clean = clean.strip()
    path  = _cache_path(f"word::{clean}", "w")
    if path.exists():
        return path
    try:
        if _edge_ok():
            _run_async(_edge(clean, path))
        elif _gtts_ok():
            _gtts(clean, path)
        else:
            return None
        return path if path.exists() else None
    except Exception as e:
        print(f"[audio_for_word] '{clean}': {e}")
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


def audio_for_row(word: str, transcription: str, lesson_id=None):
    """Smart dispatch:
    - Lessons 1, 3, 8, 16 with letter rows: pre-recorded phonemes/en/<letter>.mp3
    - Other letter rows (Aa, Bb): phoneme TTS
    - Word rows (Bad, "Flat – bad"): word TTS on the first segment
    """
    is_letter_row = bool(re.match(r"^[A-Za-z]{1,2}$", word.strip()))

    if lesson_id in PHONEME_AUDIO_LESSONS and is_letter_row:
        p = prerecorded_phoneme_path(word)
        if p:
            return p
        # fall through to TTS if file missing for this letter

    if is_letter_row:
        ipa = re.sub(r"[\[\]]", "", transcription).strip()
        return audio_for_phoneme(ipa)
    else:
        # Compound rows like "Shoo – shook" or "cat – ’cane – car – stair":
        # replace dash separators with commas so TTS speaks each segment in sequence.
        spoken = re.sub(r"\s*[–—‐‑‒\-]\s*", ", ", word.strip())
        return audio_for_word(spoken)


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
<div style="background:#13131e;border:1px solid #2a2a4a;border-radius:12px;padding:14px 18px;margin:8px 0;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <button id="pl-btn-{uid}" onclick="plToggle_{uid}()"
      style="background:#2a2a5a;color:#a0a0ff;border:1px solid #5050b0;border-radius:8px;
             padding:7px 18px;cursor:pointer;font-family:JetBrains Mono,monospace;font-size:.88rem;">
      ▶ Play All
    </button>
    <span id="pl-stat-{uid}" style="color:#6060a0;font-size:.8rem;font-family:JetBrains Mono,monospace;">ready</span>
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
    d.style.cssText='width:10px;height:10px;border-radius:50%;background:#2a2a5a;transition:.2s;';
    bar.appendChild(d);
  }}
  function dot(i,c){{
    const d=document.getElementById('dot-'+uid+'-'+i); if(!d) return;
    d.style.background = c==='active' ? '#a0a0ff' : c==='done' ? '#40c070' : '#2a2a5a';
  }}
  function stop(){{
    if(aud){{aud.pause(); aud=null;}}
    if(tmr){{clearTimeout(tmr); tmr=null;}}
    playing=false; cur=-1;
    document.getElementById('pl-btn-'+uid).textContent='▶ Play All';
    document.getElementById('pl-btn-'+uid).style.color='#a0a0ff';
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
    if(!srcs[i]){{ tmr=setTimeout(()=>playIdx(i+1), pauseMs); return; }}
    aud=new Audio(srcs[i]);
    aud.onended=()=>{{ dot(i,'done'); tmr=setTimeout(()=>playIdx(i+1), pauseMs); }};
    aud.onerror=()=>{{ tmr=setTimeout(()=>playIdx(i+1), 300); }};
    aud.play().catch(()=>{{ tmr=setTimeout(()=>playIdx(i+1), 300); }});
  }}
  window['plToggle_'+uid]=function(){{
    if(playing){{ stop(); document.getElementById('pl-stat-'+uid).textContent='stopped'; }}
    else{{
      document.getElementById('pl-btn-'+uid).textContent='■ Stop';
      document.getElementById('pl-btn-'+uid).style.color='#ff6060';
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
            color = "#40c070" if s.get("passed") else "#c04040"
            pct   = int(s.get("score", 0) * 100)
            score_html = (f'<span style="background:{"#0d2e1a" if s.get("passed") else "#2e0d0d"};'
                          f'color:{color};border-radius:5px;padding:2px 9px;'
                          f'font-family:JetBrains Mono,monospace;font-size:.78rem">{pct}%</span>')
        style = ""
        if active_idx == i:
            style = "background:#1e1e40;border-left:3px solid #6060d0;"
        html_rows += (
            f'<div class="row-ok" style="{style}">'
            f'<span style="min-width:28px;color:#4040a0;font-family:JetBrains Mono,monospace;font-size:.75rem">{i+1:02d}</span>'
            f'<span style="flex:1;color:#e0e0ff;font-size:1.05rem">{word}</span>'
            f'<span style="flex:1;color:#a0a0ff;font-family:JetBrains Mono,monospace;font-size:.9rem">{trans}</span>'
            f'{score_html}'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:#13131e;border:1px solid #222236;border-radius:12px;'
        f'overflow:hidden;margin:8px 0">{html_rows}</div>',
        unsafe_allow_html=True,
    )


def preload_lesson_audio(rows, prefix: str):
    """Cache audio paths in session_state under `{prefix}_paths` (list of str or None)."""
    key = f"{prefix}_paths"
    if key not in st.session_state:
        with st.spinner("Готуємо аудіо..."):
            paths = []
            for _, r in rows.iterrows():
                p = audio_for_row(r["word"], r["transcription"],
                                  lesson_id=int(r["lesson_id"]))
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
def load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl", sheet_name="Все уроки")
    df.columns = ["lesson_id", "row_id", "word", "transcription", "rule"]
    df["word"]          = df["word"].astype(str).str.strip()
    df["transcription"] = df["transcription"].astype(str).str.strip()
    df["rule"]          = df["rule"].fillna("").astype(str).str.strip()
    return df


st.set_page_config(page_title="Reading Practice", page_icon="📖",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0d0d14;color:#e2e2f0;}
#MainMenu,footer,header{visibility:hidden;}
.wcard{background:#13131e;border:1px solid #2a2a4a;border-radius:14px;padding:28px 20px;margin:10px 0;text-align:center;}
.wbig{font-size:3.2rem;font-weight:700;color:#f0f0ff;}
.tbig{font-size:2rem;color:#a0a0ff;font-family:'JetBrains Mono',monospace;margin-top:8px;}
.rule{background:#1a1a2e;border-left:3px solid #5050b0;border-radius:6px;padding:10px 14px;margin:8px 0;color:#8080b0;font-size:.85rem;}
.spill{font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:3px 10px;border-radius:20px;margin:2px;display:inline-block;}
.row-ok{display:flex;gap:10px;padding:8px 14px;background:#13131e;border-bottom:1px solid #1e1e30;align-items:center;}
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
            style = "background:#2a2a5a;color:#a0a0ff;border:1px solid #5050b0"
        elif s < current_step():
            style = "background:#0d2e1a;color:#40c070;border:1px solid #204030"
        elif s in REQUIRED:
            style = "background:#2e1a0d;color:#d08040;border:1px solid #704020"
        else:
            style = "background:#1a1a2e;color:#5050a0;border:1px solid #2a2a3a"
        lbl = f"{'🔒' if s in REQUIRED and s > current_step() else s}"
        pills += f'<span class="spill" style="{style}">{lbl}</span>'

    req_note = ""
    if step in REQUIRED:
        req_note = ' <span style="color:#d08040;font-size:.72rem">🔒 обов\'язковий</span>'

    st.markdown(f'<div style="margin-bottom:10px">{pills}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);'
        f'border:1px solid #2a2a4a;border-radius:14px;padding:14px 20px;margin-bottom:14px">'
        f'<div style="color:#6060c0;font-size:.75rem;font-family:JetBrains Mono,monospace">'
        f'КРОК {step} / 5{req_note}</div>'
        f'<div style="color:#f0f0ff;font-size:1.15rem;font-weight:600">{STEPS[step]}</div>'
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
        f'<div style="background:#1a1a2e;border-radius:6px;height:6px;overflow:hidden;margin:6px 0">'
        f'<div style="height:6px;background:linear-gradient(90deg,#4040c0,#6060ff);width:{val*100:.0f}%"></div></div>',
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

    # Show all letters/words with transcription
    lessons_table(rows, show_word=True, show_trans=True)

    # Preload audio paths and render autoplaylist
    paths = preload_lesson_audio(rows, "s1")
    st.markdown("Натисни **▶ Play All** — букви/слова звучатимуть з паузою 1 секунда.")
    components.html(autoplaylist_html(paths, pause_secs=1.0, uid="s1"),
                    height=130, scrolling=False)

    if st.button("Продовжити →", type="primary", use_container_width=True,
                 key="s1_done"):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2 — Прочитай слова (all on screen, mic + similarity check)
# ═══════════════════════════════════════════════════════════════════════════

def do_step2(rows: pd.DataFrame) -> bool:
    shdr(2)

    scores = st.session_state.get("s2_scores", {})

    rule_txt = next((r["rule"] for _, r in rows.iterrows() if r["rule"]), "")
    if rule_txt:
        st.markdown(f'<div class="rule">📖 {rule_txt}</div>', unsafe_allow_html=True)

    # All words + transcription on screen
    lessons_table(rows, show_word=True, show_trans=True, scores=scores)

    # Expected string: words joined by ". " so Whisper hears separate utterances
    expected = ". ".join(str(r["word"]).strip() for _, r in rows.iterrows())

    st.markdown("#### 🎙️ Прочитай вголос всі слова — запиши себе")
    audio = mic("s2")

    if not STT_OK or not SCORER_OK:
        st.caption("⚠️ Для перевірки потрібно: `pip install openai-whisper rapidfuzz`")

    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("✓ Перевірити вимову", type="primary",
                     use_container_width=True, key="s2_check"):
            if not audio:
                st.warning("Спочатку запиши аудіо!")
            elif not STT_OK or not SCORER_OK:
                st.warning("Whisper/RapidFuzz не встановлені.")
            else:
                with st.spinner("Розпізнаємо мовлення..."):
                    r = score_audio(audio, expected)
                if r:
                    # Mark every row with the same overall score (whole-recording match)
                    scores = {i: r for i in range(len(rows))}
                    st.session_state["s2_scores"] = scores
                    color = "#40c070" if r["passed"] else "#c04040"
                    st.markdown(
                        f'<div style="text-align:center;font-size:1.6rem;'
                        f'color:{color};font-weight:600">{int(r["score"]*100)}%</div>',
                        unsafe_allow_html=True,
                    )
                    st.rerun()
                else:
                    st.error("Не вдалося розпізнати аудіо.")
    with c2:
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
            f'<span style="color:{"#40c070" if v else "#c04040"};flex:1">{"✓" if v else "✗"} {rows.iloc[i]["word"]}</span>'
            f'<span style="color:#6060a0;font-family:JetBrains Mono,monospace;font-size:.78rem">{rows.iloc[i]["transcription"]}</span>'
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

    lessons_table(rows, show_word=True, show_trans=True)

    paths = preload_lesson_audio(rows, "s4")
    st.markdown("Натисни **▶ Play All** — пауза між аудіо 1 секунда.")
    components.html(autoplaylist_html(paths, pause_secs=1.0, uid="s4"),
                    height=130, scrolling=False)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Продовжити →", type="primary",
                     use_container_width=True, key="s4_done"):
            return True
    with c2:
        if st.button("⏭ Пропустити", key="s4_skip", use_container_width=True):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5 — Прочитай на час (mic-driven timer + similarity check)
# ═══════════════════════════════════════════════════════════════════════════

def do_step5(rows: pd.DataFrame) -> bool:
    shdr(5)

    # Show all words as grid
    chips = "".join(
        f'<span style="font-size:1.3rem;font-weight:600;color:#e0e0ff;'
        f'background:#13131e;border:1px solid #2a2a4a;border-radius:10px;'
        f'padding:10px 16px;margin:4px;display:inline-block">'
        f'{row["word"]}'
        f'<span style="display:block;font-size:.75rem;color:#6060a0;'
        f'font-family:JetBrains Mono,monospace">{row["transcription"]}</span></span>'
        for _, row in rows.iterrows()
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;padding:16px;'
        f'background:#0d0d14;border-radius:12px">{chips}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 🎙️ Запиши себе, поки читаєш вголос всі слова")
    audio = mic("s5")

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


# ═══════════════════════════════════════════════════════════════════════════
#  Setup screen
# ═══════════════════════════════════════════════════════════════════════════

def render_setup(df: pd.DataFrame):
    st.markdown("""
    <div style="text-align:center;padding:36px 0 20px">
      <div style="font-size:3rem">📖</div>
      <h1 style="color:#f0f0ff;font-weight:600;margin:10px 0 4px">Reading Practice</h1>
      <p style="color:#606090">English Phonics · 80 уроків · IPA озвучка</p>
    </div>""", unsafe_allow_html=True)

    if not _edge_ok() and not _gtts_ok():
        st.error("⚠️ Встанови аудіо бібліотеку:\n\n`pip install edge-tts`\n\nабо\n\n`pip install gtts`")

    lessons = sorted(df["lesson_id"].unique())
    c1, c2  = st.columns(2)
    with c1:
        lesson_id = st.selectbox(
            "📚 Урок", lessons,
            format_func=lambda x: f"Урок {x} — {len(df[df['lesson_id']==x])} рядків",
        )
    with c2:
        user_id = st.text_input("👤 Ім'я", value="student1")

    rows = df[df["lesson_id"] == lesson_id].reset_index(drop=True)

    st.markdown(f"**{len(rows)} слів/рядків у цьому уроці:**")
    preview = "".join(
        '<div style="display:flex;gap:14px;padding:8px 14px;background:#13131e;'
        'border-bottom:1px solid #1e1e30;align-items:center">'
        f'<span style="min-width:24px;color:#4040a0;font-family:JetBrains Mono,monospace;font-size:.75rem">{i+1:02d}</span>'
        f'<span style="flex:1;font-size:1rem;color:#e0e0ff">{row["word"]}</span>'
        f'<span style="color:#a0a0ff;font-family:JetBrains Mono,monospace;font-size:.85rem">{row["transcription"]}</span>'
        + ('<span style="color:#5050a0;font-size:.75rem;margin-left:8px">'
           + row["rule"][:40] + '...</span>' if len(row["rule"]) > 5 else '')
        + '</div>'
        for i, (_, row) in enumerate(rows.iterrows())
    )
    st.markdown(
        f'<div style="border-radius:10px;overflow:hidden;max-height:280px;overflow-y:auto">'
        f'{preview}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    if st.button("▶ Почати урок", type="primary", use_container_width=True):
        st.session_state["r_lesson"] = int(lesson_id)
        st.session_state["r_user"]   = user_id
        st.session_state["r_rows"]   = rows
        st.session_state["r_step"]   = 1
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if not DB_PATH.exists():
        st.error(
            f"**Файл не знайдено:** `{DB_PATH}`\n\n"
            "Скопіюй Excel файл у `data/reading_lessons.xlsx`."
        )
        st.stop()

    df = load(str(DB_PATH))

    if "r_step" not in st.session_state:
        render_setup(df)
        return

    step = st.session_state["r_step"]
    rows = st.session_state["r_rows"]

    with st.sidebar:
        all_l = sorted(df["lesson_id"].unique())
        lid   = st.session_state.get("r_lesson", 1)
        pct   = round((lid - 1) / max(len(all_l), 1) * 100, 1)
        st.markdown(
            f'<div style="font-size:.75rem;color:#5050a0;font-family:JetBrains Mono,monospace">'
            f'Урок {lid} / {len(all_l)} · {pct}%</div>'
            f'<div style="background:#1a1a2e;border-radius:6px;height:6px;overflow:hidden">'
            f'<div style="height:6px;background:linear-gradient(90deg,#4040c0,#6060ff);'
            f'width:{pct}%"></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Крок {step}/5** — {STEPS.get(step,'')}")
        st.markdown("---")
        if st.button("🏠 Головне меню"):
            clear_all()
            st.rerun()

    if step > 5:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0d2e1a,#1a1a2e);
             border:1px solid #304030;border-radius:16px;padding:40px;text-align:center">
          <div style="font-size:3rem">🎉</div>
          <h2 style="color:#f0f0ff">Урок завершено!</h2>
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Повторити", type="primary", use_container_width=True):
                clear_step_state()
                st.session_state["r_step"] = 1
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
    elif step not in REQUIRED:
        st.markdown("---")
        if st.button(f"⏭ Пропустити крок {step}", key=f"skip_global_{step}"):
            clear_step_state()
            st.session_state["r_step"] = step + 1
            st.rerun()


if __name__ == "__main__":
    main()
