"""
IMLLS — Intelligent Multilingual Language Learning System
Lesson-based flow: 7 steps, full lesson visible at once.
JS MediaRecorder for in-browser audio capture.

Run: streamlit run app.py
"""
import sys, base64, random, time, json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

LESSON_IMG_DIR = ROOT / "static" / "lesson_images"

@st.cache_data(show_spinner=False)
def _lesson_img_b64(lesson_id: int) -> str | None:
    """Return base64 data-URL for a lesson illustration, or None if missing."""
    p = LESSON_IMG_DIR / f"lesson_{lesson_id:03d}.png"
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

def _show_lesson_image(lesson_id: int, max_width_px: int = 340):
    """Show centered lesson illustration if the file exists."""
    p = LESSON_IMG_DIR / f"lesson_{lesson_id:03d}.png"
    if not p.exists():
        return
    # Center by putting image in a middle column
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(str(p), use_container_width=True)

from engine.loader  import (load_phrases, get_lesson, get_available_lessons,
                            get_lesson_topics as get_grammar_topics,
                            TTS_LANG, WHISPER_LANG)
from engine.vocab_loader import (
    load_vocab, get_vocab_lesson, get_available_vocab_lessons, get_lesson_topics,
)
from engine.session import LessonSession
from engine.scorer  import evaluate
from engine.tts     import get_audio_path
from engine.stt      import transcribe_bytes, whisper_available
from engine.logger  import SessionLogger, get_last_lesson, get_progress, save_progress
from engine.gec import correct as gec_correct, gec_available
from engine.gamification import on_step_complete, on_lesson_complete, sidebar_widget
from engine.picker import _render_flat_wave_nav, _render_vocab_nav
from engine import i18n
from engine.character_widget import show_character

try:
    from engine.adaptive import AdaptiveEngine as _AdaptiveEngine, FULL_SEQUENCE as _FULL_SEQ
    _ADAPTIVE_AVAILABLE = True
except Exception:
    _ADAPTIVE_AVAILABLE = False
    _FULL_SEQ = [1, 2, 3, 4, 5, 6, 7, 8]

# Prefer the workbook with lesson titles (topic_en / topic_uk columns) if it
# exists; otherwise fall back to the original. Both have identical phrase data.
DB_PATH_TITLED = ROOT / "data" / "imlls_database_with_titles.xlsx"
DB_PATH_PLAIN  = ROOT / "data" / "imlls_database.xlsx"
DB_PATH        = DB_PATH_TITLED if DB_PATH_TITLED.exists() else DB_PATH_PLAIN
VOCAB_DB_PATH  = ROOT / "data" / "vocabulary.xlsx"
LANGUAGES      = ["English", "Ukrainian", "Spanish", "Korean"]

# st.set_page_config is set up by main_app.py when used as a launcher.
# When this file is run directly, set it here too.
try:
    st.set_page_config(page_title="IMLLS", page_icon="🗣️",
                       layout="wide", initial_sidebar_state="collapsed")
except Exception:
    pass  # Already set by main_app.py


# ═══════════════════════════════════════════════════════════════════════════
# Module dispatch — same 8-step flow for grammar and vocabulary
# ═══════════════════════════════════════════════════════════════════════════
def _module_config(module: str) -> dict:
    """Return loader dispatch + UI labels for the chosen practice module."""
    if module == "vocab":
        return {
            "db_path":     VOCAB_DB_PATH,
            "load":        load_vocab,
            "get_lesson":  get_vocab_lesson,
            "get_lessons": get_available_vocab_lessons,
            "topics":      get_lesson_topics,
            "label":       "Vocabulary",
            "icon":        "📖",
            "lang_suffix": "vocab",
            "lesson_word": "Topic",
        }
    if module == "custom":
        # User-created lessons. Data is loaded on demand by custom_app,
        # so the "load"/"get_lessons" callables are intentionally None —
        # they're never called for this module (custom_app pushes the
        # session into st.session_state before grammar.main() runs).
        return {
            "db_path":     None,
            "load":        None,
            "get_lesson":  None,
            "get_lessons": None,
            "topics":      None,
            "label":       "My Phrases",
            "icon":        "📝",
            "lang_suffix": "custom",
            "lesson_word": "Lesson",
        }
    return {
        "db_path":     DB_PATH,
        "load":        load_phrases,
        "get_lesson":  get_lesson,
        "get_lessons": get_available_lessons,
        # Grammar topics come from the loaded DataFrame (optional topic_en/topic_uk)
        "topics":      "from_df",
        "label":       "Grammar",
        "icon":        "🗣️",
        "lang_suffix": "grammar",
        "lesson_word": "Lesson",
    }


def _current_module() -> str:
    return st.session_state.get("practice_module", "grammar")

# ═══════════════════════════════════════════════════════════════════════════
# Audio input — st.audio_input (Streamlit 1.31+) with fallback
# ═══════════════════════════════════════════════════════════════════════════

def audio_input(uid: str, label: str = "🎙️ Record") -> bytes | None:
    """
    Uses st.audio_input (built-in mic recorder, no JS iframe needed).
    Falls back to file_uploader if st.audio_input is not available.
    Returns raw audio bytes or None.
    """
    if hasattr(st, "audio_input"):
        recorded = st.audio_input(label, key=f"mic_{uid}")
        if recorded:
            return recorded.read()
        return None
    else:
        # Fallback for older Streamlit versions
        st.caption("⚠️ Upgrade Streamlit to 1.31+ for built-in mic recording.")
        f = st.file_uploader("Upload audio", type=["webm","wav","mp3","ogg","m4a"],
                             key=f"up_{uid}", label_visibility="collapsed")
        return f.read() if f else None





def autoplaylist_html(audio_paths, pause_secs):
    """JS component: plays a list of MP3s sequentially with custom pauses."""
    srcs = []
    for p in audio_paths:
        if p and Path(p).exists():
            with open(p, "rb") as f:
                srcs.append("data:audio/mp3;base64," + base64.b64encode(f.read()).decode())
        else:
            srcs.append("")
    srcs_js   = str(srcs).replace("'", '"')
    pauses_js = str([round(s, 2) for s in pause_secs])
    n = len(srcs)
    return f"""
<div style="background:#FFFFFF;border:1px solid #E8E2D8;border-radius:12px;padding:14px 18px;margin:8px 0;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <button id="pl-btn" onclick="plToggle()"
      style="background:#ECEBFB;color:#4F46E5;border:1px solid #4F46E5;border-radius:8px;
             padding:7px 18px;cursor:pointer;font-family:JetBrains Mono,monospace;font-size:.88rem;">
      ▶ Play All
    </button>
    <span id="pl-stat" style="color:#7A7390;font-size:.8rem;font-family:JetBrains Mono,monospace;">ready</span>
  </div>
  <div id="pl-bar" style="margin-top:10px;display:flex;gap:4px;flex-wrap:wrap;"></div>
</div>
<script>
(function(){{
  const srcs={srcs_js}, pauses={pauses_js}, n={n};
  let cur=-1, playing=false, aud=null, tmr=null;
  const bar=document.getElementById('pl-bar');
  for(let i=0;i<n;i++){{const d=document.createElement('div');d.id='dot-'+i;
    d.style.cssText='width:10px;height:10px;border-radius:50%;background:#ECEBFB;transition:.2s;';
    bar.appendChild(d);}}
  function dot(i,c){{const d=document.getElementById('dot-'+i);if(!d)return;
    d.style.background=c==='active'?'#4F46E5':c==='done'?'#1FB888':'#ECEBFB';}}
  function stop(){{if(aud){{aud.pause();aud=null;}}if(tmr){{clearTimeout(tmr);tmr=null;}}
    playing=false;cur=-1;document.getElementById('pl-btn').textContent='▶ Play All';
    document.getElementById('pl-btn').style.color='#4F46E5';
    document.getElementById('pl-stat').textContent='stopped';
    for(let i=0;i<n;i++)dot(i,'');}}
  function playIdx(i){{if(i>=n){{stop();document.getElementById('pl-stat').textContent='done ✓';return;}}
    cur=i;playing=true;for(let j=0;j<i;j++)dot(j,'done');dot(i,'active');
    document.getElementById('pl-stat').textContent='phrase '+(i+1)+'/'+n;
    if(!srcs[i]){{tmr=setTimeout(()=>playIdx(i+1),pauses[i]*1000);return;}}
    aud=new Audio(srcs[i]);
    aud.onended=()=>{{dot(i,'done');tmr=setTimeout(()=>playIdx(i+1),pauses[i]*1000);}};
    aud.onerror=()=>{{tmr=setTimeout(()=>playIdx(i+1),500);}};
    aud.play().catch(()=>{{tmr=setTimeout(()=>playIdx(i+1),500);}});}}
  window.plToggle=function(){{if(playing){{stop();}}else{{
    document.getElementById('pl-btn').textContent='■ Stop';
    document.getElementById('pl-btn').style.color='#FF7B6B';playIdx(0);}}}};
}})();
</script>
"""

def autoplaylist_html_with_highlight(audio_paths, pause_secs, uid="pl"):
    """Autoplaylist that also updates a URL param so Python can highlight active phrase."""
    srcs = []
    for p in audio_paths:
        if p and Path(p).exists():
            with open(p, "rb") as f:
                srcs.append("data:audio/mp3;base64," + base64.b64encode(f.read()).decode())
        else:
            srcs.append("")
    srcs_js   = str(srcs).replace("'", '"')
    pauses_js = str([round(s, 2) for s in pause_secs])
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
  const srcs={srcs_js},pauses={pauses_js},n={n},uid='{uid}';
  let cur=-1,playing=false,aud=null,tmr=null;
  const bar=document.getElementById('pl-bar-'+uid);
  for(let i=0;i<n;i++){{const d=document.createElement('div');d.id='dot-'+uid+'-'+i;
    d.style.cssText='width:10px;height:10px;border-radius:50%;background:#ECEBFB;transition:.2s;';
    bar.appendChild(d);}}
  function dot(i,c){{const d=document.getElementById('dot-'+uid+'-'+i);if(!d)return;
    d.style.background=c==='active'?'#4F46E5':c==='done'?'#1FB888':'#ECEBFB';}}
  function stop(){{if(aud){{aud.pause();aud=null;}}if(tmr){{clearTimeout(tmr);tmr=null;}}
    playing=false;cur=-1;
    document.getElementById('pl-btn-'+uid).textContent='▶ Play All';
    document.getElementById('pl-btn-'+uid).style.color='#4F46E5';
    document.getElementById('pl-stat-'+uid).textContent='done ✓';
    for(let i=0;i<n;i++)dot(i,'done');}}
  function playIdx(i){{
    if(i>=n){{stop();return;}}
    cur=i;playing=true;
    for(let j=0;j<i;j++)dot(j,'done');dot(i,'active');
    document.getElementById('pl-stat-'+uid).textContent='▶ phrase '+(i+1)+'/'+n;
    // Notify parent frame of active index for row highlighting
    try{{window.parent.postMessage({{type:'imlls_highlight',uid:uid,idx:i}},'*');}}catch(e){{}}
    if(!srcs[i]){{tmr=setTimeout(()=>playIdx(i+1),pauses[i]*1000);return;}}
    aud=new Audio(srcs[i]);
    aud.onended=()=>{{dot(i,'done');tmr=setTimeout(()=>playIdx(i+1),pauses[i]*1000);}};
    aud.onerror=()=>{{tmr=setTimeout(()=>playIdx(i+1),500);}};
    aud.play().catch(()=>{{tmr=setTimeout(()=>playIdx(i+1),500);}});}}
  window['plToggle_'+uid]=function(){{
    if(playing){{stop();document.getElementById('pl-stat-'+uid).textContent='stopped';}}
    else{{document.getElementById('pl-btn-'+uid).textContent='■ Stop';
          document.getElementById('pl-btn-'+uid).style.color='#FF7B6B';playIdx(0);}}}};
}})();
</script>
"""

def autoplaylist_with_table(phrases, audio_paths, pause_secs, uid="pl",
                            show_native=True, show_target=True):
    """All-in-one component: phrase table + audio player inside the same iframe.
    The currently playing phrase row gets a bright highlight so the user can
    see exactly which phrase is being spoken."""
    import json as _json

    # Build base64-encoded audio sources
    srcs = []
    for p in audio_paths:
        if p and Path(p).exists():
            with open(p, "rb") as f:
                srcs.append("data:audio/mp3;base64," + base64.b64encode(f.read()).decode())
        else:
            srcs.append("")
    srcs_js   = _json.dumps(srcs)
    pauses_js = _json.dumps([round(s, 2) for s in pause_secs])
    n = len(srcs)

    # Build the phrase table HTML rows
    def _esc(t):
        # Minimal HTML escape so the rows can't break the markup
        return (str(t).replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;"))

    rows_html = ""
    for i, p in enumerate(phrases):
        nat = _esc(p["native"]) if show_native else "—"
        tgt = _esc(p["target"]) if show_target else "—"
        rows_html += (
            f'<div class="ph-row" id="row-{uid}-{i}">'
            f'  <span class="ph-num">{i+1:02d}</span>'
            f'  <span class="ph-nat">{nat}</span>'
            f'  <span class="ph-tgt">{tgt}</span>'
            f'</div>'
        )

    return f"""
<style>
  .ph-table {{
    background:#FFFFFF; border:1px solid #DDD6CA; border-radius:12px;
    overflow:hidden; margin:10px 0;
  }}
  .ph-row {{
    display:flex; align-items:center; padding:11px 18px;
    border-bottom:1px solid #E8E2D8; gap:14px;
    transition: background .15s, border-left-color .15s;
    border-left:3px solid transparent;
  }}
  .ph-row:last-child {{ border-bottom:none; }}
  .ph-num {{ font-family:'JetBrains Mono',monospace; color:#2E27A8;
            font-size:.73rem; min-width:26px; }}
  .ph-nat {{ color:#7A7390; flex:1; font-size:.93rem; }}
  .ph-tgt {{ color:#4B4564; flex:1; font-size:.93rem; font-weight:500; }}
  .ph-row.active {{
    background:#ECEBFB; border-left-color:#4F46E5;
  }}
  .ph-row.active .ph-nat {{ color:#1B1730; }}
  .ph-row.active .ph-tgt {{ color:#1B1730; font-weight:600; }}
  .ph-row.done {{ background:#F4F0EB; }}
  .ph-row.done .ph-tgt {{ color:#0E6E50; }}
  .pl-wrap {{
    background:#FFFFFF; border:1px solid #E8E2D8; border-radius:12px;
    padding:14px 18px; margin:8px 0;
  }}
  .pl-btn {{
    background:#ECEBFB; color:#4F46E5; border:1px solid #4F46E5;
    border-radius:8px; padding:7px 18px; cursor:pointer;
    font-family:'JetBrains Mono',monospace; font-size:.88rem;
  }}
  .pl-stat {{
    color:#7A7390; font-size:.8rem; font-family:'JetBrains Mono',monospace;
    margin-left:12px;
  }}
  .pl-bar {{ margin-top:10px; display:flex; gap:4px; flex-wrap:wrap; }}
  .pl-dot {{
    width:10px; height:10px; border-radius:50%;
    background:#ECEBFB; transition:.2s;
  }}
</style>

<!-- Player FIRST so mobile users see the Play All button without scrolling -->
<div class="pl-wrap">
  <button id="pl-btn-{uid}" class="pl-btn" onclick="plToggle_{uid}()">▶ Play All</button>
  <span id="pl-stat-{uid}" class="pl-stat">ready</span>
  <div id="pl-bar-{uid}" class="pl-bar"></div>
</div>

<div class="ph-table">{rows_html}</div>

<script>
(function(){{
  const srcs={srcs_js}, pauses={pauses_js}, n={n}, uid='{uid}';
  let cur=-1, playing=false, aud=null, tmr=null;

  // Build dots
  const bar = document.getElementById('pl-bar-'+uid);
  for (let i=0; i<n; i++) {{
    const d = document.createElement('div');
    d.id = 'dot-'+uid+'-'+i;
    d.className = 'pl-dot';
    bar.appendChild(d);
  }}
  function dot(i, c) {{
    const d = document.getElementById('dot-'+uid+'-'+i);
    if (!d) return;
    d.style.background = c==='active' ? '#4F46E5'
                       : c==='done'   ? '#1FB888' : '#ECEBFB';
  }}

  // Highlight rows
  function setRow(i, cls) {{
    const r = document.getElementById('row-'+uid+'-'+i);
    if (!r) return;
    if (cls === 'clear') {{ r.classList.remove('active','done'); return; }}
    r.classList.remove('active','done');
    r.classList.add(cls);
  }}
  function clearAllRows() {{
    for (let i=0; i<n; i++) setRow(i, 'clear');
  }}

  function ensureAud() {{
    // Create ONE Audio element only inside a user-gesture handler.
    // iOS Safari grants playback permission to elements unlocked by a tap;
    // we reuse this element for every track so the permission persists.
    if (aud) return;
    aud = new Audio();
    aud.preload = 'auto';
    aud.addEventListener('ended', function() {{
      var i = cur;
      dot(i, 'done'); setRow(i, 'done');
      tmr = setTimeout(function() {{ playIdx(i + 1); }}, pauses[i] * 1000);
    }});
    aud.addEventListener('error', function() {{
      tmr = setTimeout(function() {{ playIdx(cur + 1); }}, 500);
    }});
  }}

  function stop() {{
    if (aud) {{ try {{ aud.pause(); }} catch(e) {{}} }}
    if (tmr) {{ clearTimeout(tmr); tmr = null; }}
    playing = false; cur = -1;
    document.getElementById('pl-btn-'+uid).textContent = '▶ Play All';
    document.getElementById('pl-btn-'+uid).style.color = '#4F46E5';
  }}

  function playIdx(i) {{
    if (i >= n) {{
      stop();
      document.getElementById('pl-stat-'+uid).textContent = 'done ✓';
      for (let j=0; j<n; j++) {{ dot(j,'done'); setRow(j,'done'); }}
      return;
    }}
    cur = i; playing = true;
    // Mark previous as done, current as active
    for (let j=0; j<i; j++) {{ dot(j,'done'); setRow(j,'done'); }}
    dot(i, 'active'); setRow(i, 'active');
    document.getElementById('pl-stat-'+uid).textContent = '▶ phrase ' + (i+1) + ' / ' + n;

    if (!srcs[i]) {{
      tmr = setTimeout(function() {{ playIdx(i+1); }}, pauses[i] * 1000);
      return;
    }}
    aud.src = srcs[i];
    var p = aud.play();
    if (p && typeof p.catch === 'function') {{
      p.catch(function() {{
        tmr = setTimeout(function() {{ playIdx(i+1); }}, 500);
      }});
    }}
  }}

  window['plToggle_'+uid] = function() {{
    if (playing) {{
      stop();
      document.getElementById('pl-stat-'+uid).textContent = 'stopped';
    }} else {{
      ensureAud();  // must run inside this user-gesture click handler
      clearAllRows();
      for (let i=0; i<n; i++) dot(i, '');
      document.getElementById('pl-btn-'+uid).textContent = '■ Stop';
      document.getElementById('pl-btn-'+uid).style.color = '#FF7B6B';
      playIdx(0);
    }}
  }};
}})();
</script>
"""


def phrase_pause(phrase):
    words = len(phrase.split())
    return min(max(round(words / 2, 1), 1.5), 4.0)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def play(path: str, auto=False):
    if not path or not Path(path).exists(): return
    with open(path,"rb") as f: d=base64.b64encode(f.read()).decode()
    a = "autoplay" if auto else ""
    st.markdown(f'<audio controls {a}><source src="data:audio/mp3;base64,{d}" type="audio/mp3"></audio>',
                unsafe_allow_html=True)


def phrase_table(phrases, show_native=True, show_target=True, scores=None, highlight=None, active_idx=None):
    html = ""
    for i, p in enumerate(phrases):
        num  = f'<span class="pnum">{i+1:02d}</span>'
        nat  = f'<span class="pnat">{p["native"]}</span>'  if show_native  else '<span class="phide">—</span>'
        tgt  = f'<span class="ptgt">{p["target"]}</span>'  if show_target  else '<span class="phide">—</span>'
        sc   = ""
        if scores and i in scores:
            s   = scores[i]
            cls = "spass" if s["passed"] else "sfail"
            sc  = f'<span class="{cls}">{int(s["score"]*100)}%</span>'
        # active_idx = currently playing phrase (bright highlight)
        # highlight  = legacy dimmer highlight
        if active_idx == i:
            row_cls = ' class="prow prow-active"'
        elif highlight == i:
            row_cls = ' class="prow" style="background:var(--mova-surface-2)"'
        else:
            row_cls = ' class="prow"'
        html += f'<div{row_cls}>{num}{nat}{tgt}{sc}</div>'
    st.markdown(f'<div class="ptable">{html}</div>', unsafe_allow_html=True)


def step_hdr(step, title=None, desc=None, total=8, show_image=False):
    """Render the step header.

    Title and hint are read from engine.i18n based on the current session's
    native language. Adaptive step sequence (if active) is read from
    session_state so every step function gets correct pill rendering
    without changing its own signature.
    """
    # Look up the user's native language from the active session
    sess = st.session_state.get("session")
    native_lang = sess.state.native_lang if sess else "English"

    if title is None:
        title = i18n.get(native_lang, "titles", step)
    if desc is None:
        desc = i18n.get(native_lang, "hints", step)

    icon       = i18n.step_icon(step)
    s_type     = i18n.step_type(step)
    step_label = i18n.get(native_lang, "step_label")
    req_word   = i18n.get(native_lang, "required")

    # ── Adaptive-aware pill rendering ──────────────────────────────────────
    adp_steps = st.session_state.get("_adaptive_steps")
    adp_idx   = st.session_state.get("_adaptive_idx", 0)

    if adp_steps:
        seq       = adp_steps
        adp_total = len(seq)
        pos_label = adp_idx + 1        # 1-based position shown to user

        # Deduplicate while preserving order (handles EXTRA_REPEAT)
        seen_s: set = set()
        unique_seq = [s for s in seq if not (s in seen_s or seen_s.add(s))]  # type: ignore[func-returns-value]

        def _pcls(s: int) -> str:
            occurrences = [i for i, x in enumerate(seq) if x == s]
            if s == step and any(i == adp_idx for i in occurrences):
                return "pill-active"
            if max(occurrences) < adp_idx:
                return "pill-done"
            if s in REQUIRED_STEPS:
                return "pill-required"
            return ""

        def _plbl(s: int) -> str:
            occurrences = [i for i, x in enumerate(seq) if x == s]
            pending = max(occurrences) >= adp_idx and not any(i == adp_idx for i in occurrences)
            return "🔒" if (pending and s in REQUIRED_STEPS) else str(s)

        pills = "".join(
            f'<span class="pill {_pcls(s)}">{_plbl(s)}</span>'
            for s in unique_seq
        )
    else:
        adp_total = total
        pos_label = step
        pills = "".join(
            f'<span class="pill {"pill-active" if s==step else "pill-done" if s<step else "pill-required" if s in REQUIRED_STEPS and s>step else ""}">'
            f'{"🔒" if s in REQUIRED_STEPS and s > step else s}</span>'
            for s in range(1, total + 1)
        )

    req_note = (f' <span style="color:var(--mova-amber-ink);font-size:.72rem;">🔒 {req_word}</span>'
                if step in REQUIRED_STEPS else '')

    st.markdown(f"""
    <div class="step-header step-type-{s_type}">
      <div class="step-pills">{pills}</div>
      <div class="step-num">{step_label} {pos_label} / {adp_total}{req_note}</div>
      <div class="step-icon-row">
        <span class="step-icon-big">{icon}</span>
        <div style="flex:1">
          <div class="step-title">{title}</div>
          <div class="step-desc">{desc}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Show lesson illustration if requested
    if show_image and sess:
        _show_lesson_image(sess.state.lesson_id)


def _audio_duration_ms(audio_bytes: bytes) -> int:
    """Extract audio duration from WAV/WebM bytes. Returns ms, or 0 on failure."""
    try:
        import io, wave
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            return int(wf.getnframes() / wf.getframerate() * 1000)
    except Exception:
        pass
    try:
        # Fallback: estimate from file size (~16kHz mono webm)
        return max(0, int(len(audio_bytes) / 32000 * 1000))
    except Exception:
        return 0


def do_score(session: LessonSession, audio: bytes, expected: str,
             lang: str, step: int, phrase_id: int = 0) -> dict | None:
    """Transcribe audio, score against expected, log with audio duration."""
    if not audio: return None
    if not whisper_available():
        st.warning("Whisper not installed — install `openai-whisper` for voice scoring.")
        return None
    duration_ms = _audio_duration_ms(audio)
    with st.spinner("Transcribing…"):
        text = transcribe_bytes(audio, language=lang)
    # Pass duration_ms so logger records actual recording length, not wall-clock
    result = session.score(text, expected, step=step, phrase_id=phrase_id,
                           duration_ms=duration_ms if duration_ms > 0 else None)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Read all phrases (both languages visible)
# ═══════════════════════════════════════════════════════════════════════════
def step1(session: LessonSession, tts_lang, wh_lang):
    session.start_step(1)
    step_hdr(1, show_image=True)
    phrases = session.phrases()
    scores  = st.session_state.get("s1_scores", {})

    # Mic at the top so mobile users don't need to scroll past the phrase list
    audio = audio_input("s1")

    if audio and st.button("Submit & Check", type="primary",
                           use_container_width=True, key="s1_submit"):
        full = ". ".join(p["target"] for p in phrases)
        r    = do_score(session, audio, full, wh_lang, step=1, phrase_id=0)
        if r:
            scores = {i: r for i in range(len(phrases))}
            st.session_state["s1_scores"] = scores
            st.markdown(
                f"{'✓' if r['passed'] else '✗'} **{int(r['score']*100)}%** — "
                f"`{r['transcribed']}`"
            )

    # Character above phrases — shown once per step entry, not on button-click rerun
    if "s1_char_shown" not in st.session_state:
        st.session_state["s1_char_shown"] = True
        show_character("natalia", "motivation")

    # Phrase list
    phrase_table(phrases, show_native=True, show_target=True, scores=scores)

    # Next button below phrases
    if st.button("Next →", use_container_width=True, key="s1_next"):
        st.session_state.pop("s1_char_shown", None)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Listen & Repeat (auto-playlist with pause, both languages visible)
# ═══════════════════════════════════════════════════════════════════════════
def step2(session: LessonSession, tts_lang, wh_lang):
    session.start_step(2)
    step_hdr(2, show_image=True)
    phrases = session.phrases()
    paths   = [get_audio_path(p["target"], tts_lang) for p in phrases]
    pauses  = [phrase_pause(p["target"]) for p in phrases]

    # Caption removed — step_hdr shows the instruction in user's native language.
    # Combined table + player: the row of the currently playing phrase is highlighted
    height = 200 + 48 * len(phrases)
    components.html(
        autoplaylist_with_table(phrases, paths, pauses, uid="s2",
                                show_native=True, show_target=True),
        height=height, scrolling=True,
    )

    if st.button("Continue to Step 3 →", type="primary", use_container_width=True):
        st.session_state.pop("s2_active", None)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Listen & Match (hear target → pick correct native from shuffled list)
# ═══════════════════════════════════════════════════════════════════════════
def step3(session: LessonSession, tts_lang, wh_lang):
    session.start_step(3)
    step_hdr(3, show_image=True)
    phrases = session.phrases()

    if "s3_idx" not in st.session_state:
        st.session_state["s3_idx"]    = 0
        st.session_state["s3_scores"] = {}
        # pre-shuffle all choices per phrase
        for i in range(len(phrases)):
            opts = [p["native"] for p in phrases]
            random.shuffle(opts)
            st.session_state[f"s3_opts_{i}"] = opts

    idx    = st.session_state["s3_idx"]
    scores = st.session_state["s3_scores"]

    # Answered so far
    if scores:
        done_html = "".join(
            f'<div class="prow"><span class="pnum">{i+1:02d}</span>'
            f'<span style="color:{"var(--mova-mint-ink)" if v else "var(--mova-coral-ink)"};flex:1">'
            f'{"✓" if v else "✗"} {phrases[i]["target"]}</span></div>'
            for i, v in sorted(scores.items())
        )
        st.markdown(f'<div class="ptable">{done_html}</div>', unsafe_allow_html=True)

    if idx < len(phrases):
        p  = phrases[idx]
        ap = get_audio_path(p["target"], tts_lang)
        st.markdown(f"---\n**Phrase {idx+1} — listen and choose:**")
        if ap:
            # Use a unique key per phrase index so audio reloads on each new phrase
            with open(ap, "rb") as f: d = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<audio controls autoplay key="{idx}" style="width:100%;border-radius:8px;margin:4px 0">'
                f'<source src="data:audio/mp3;base64,{d}" type="audio/mp3"></audio>',
                unsafe_allow_html=True,
            )

        st.markdown("**Select the correct translation:**")
        for ci, choice in enumerate(st.session_state[f"s3_opts_{idx}"]):
            if st.button(choice, key=f"s3_{idx}_{ci}", use_container_width=True):
                correct = (choice == p["native"])
                scores[idx] = correct
                st.session_state["s3_scores"] = scores
                st.session_state["s3_idx"]    = idx + 1
                session.score(
                    p["native"] if correct else choice,
                    p["native"],
                    step=3,
                    phrase_id=int(p.get("phrase_id", idx)),
                )
                if correct:
                    st.success("✓ Correct!")
                else:
                    st.error(f"✗ Wrong — answer: **{p['native']}**")
                time.sleep(0.4)
                st.rerun()

    if idx >= len(phrases):
        ok = sum(1 for v in scores.values() if v)
        st.success(f"Done! {ok}/{len(phrases)} correct.")
        show_character("sophie", "on_lesson_complete")
        if st.button("Continue to Step 4 →", type="primary"):
            for k in ["s3_idx","s3_scores"] + [f"s3_opts_{i}" for i in range(len(phrases))]:
                st.session_state.pop(k, None)
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Speed Reading (target only, timer starts on Record)
# ═══════════════════════════════════════════════════════════════════════════
def step4(session: LessonSession, tts_lang, wh_lang):
    session.start_step(4)
    step_hdr(4, show_image=True)
    phrases = session.phrases()

    # Mic at the top so mobile users don't need to scroll past the phrase list
    audio = audio_input("s4")

    # Submit button appears only after the user records audio
    if audio and st.button("Submit & Score", type="primary",
                           use_container_width=True, key="s4_submit"):
        duration_s = max(1, round(_audio_duration_ms(audio) / 1000))
        full = ". ".join(p["target"] for p in phrases)
        r = do_score(session, audio, full, wh_lang, step=4, phrase_id=0)
        if r:
            st.session_state["s4_result"] = {
                "time":  duration_s,
                "score": r["score"],
                "text":  r["transcribed"],
            }

    if "s4_result" in st.session_state:
        res = st.session_state["s4_result"]
        st.success(f"🏁 Reading speed: **{res['time']}s** · **{int(res['score']*100)}%** match")
        st.caption(f"Transcribed: {res['text']}")

    # Phrase list
    phrase_table(phrases, show_native=False, show_target=True)

    # Done button below phrases
    if st.button("Done →", use_container_width=True, key="s4_done"):
        st.session_state.pop("s4_result", None)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Shadowing (auto-playlist, see native only, repeat target aloud)
# ═══════════════════════════════════════════════════════════════════════════
def step5(session: LessonSession, tts_lang, wh_lang):
    session.start_step(5)
    step_hdr(5, show_image=True)
    phrases = session.phrases()
    paths   = [get_audio_path(p["target"], tts_lang) for p in phrases]
    pauses  = [phrase_pause(p["target"]) for p in phrases]

    # Caption removed — step_hdr shows the instruction in user's native language.
    # Combined table + player: only native is shown, target row is highlighted as it plays
    height = 200 + 48 * len(phrases)
    components.html(
        autoplaylist_with_table(phrases, paths, pauses, uid="s5",
                                show_native=True, show_target=False),
        height=height, scrolling=True,
    )

    if st.button("Continue to Step 6 →", type="primary", use_container_width=True):
        st.session_state.pop("s5_active", None)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Active Translation (all phrases at once, single recording)
# ═══════════════════════════════════════════════════════════════════════════
def step6(session: LessonSession, tts_lang, wh_lang):
    session.start_step(6)
    step_hdr(6, show_image=True)
    phrases = session.phrases()
    scores  = st.session_state.get("s6_scores", {})

    # Mic at the top so mobile users don't need to scroll past the phrase list
    audio = audio_input("s6")

    if audio and st.button("Submit & Score", type="primary", use_container_width=True, key="s6_submit"):
        full = ". ".join(p["target"] for p in phrases)
        r = do_score(session, audio, full, wh_lang, step=6, phrase_id=0)
        if r:
            scores = {i: r for i in range(len(phrases))}
            st.session_state["s6_scores"] = scores
            st.session_state["s6_done"] = True
            st.markdown(f"{'✓' if r['passed'] else '✗'} **{int(r['score']*100)}%** — `{r['transcribed']}`")

    # Character above phrases — shown once per step entry, not on button-click rerun
    if "s6_char_shown" not in st.session_state:
        st.session_state["s6_char_shown"] = True
        show_character("mark", "motivation")

    # Phrase list (native shown, target hidden)
    phrase_table(phrases, show_native=True, show_target=False, scores=scores)

    # Next button below phrases
    if st.button("Next →", use_container_width=True, key="s6_next"):
        st.session_state["s6_idx"] = 0
        st.session_state.pop("s6_char_shown", None)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — Speed Translation (native only, timer, translate all)
# ═══════════════════════════════════════════════════════════════════════════
# Step 7 pass criteria — also used to inform the user up-front
_S7_SECONDS_PER_TWO_WORDS = 1.5   # ≤ 1 sec per 2 target-language words
_S7_MIN_SIMILARITY        = 0.80  # ≥ 80% similarity


def step7(session: LessonSession, tts_lang, wh_lang):
    session.start_step(7)
    step_hdr(7, show_image=True)
    phrases = session.phrases()

    # ── Pass criteria for this lesson ──
    # Speed target: 1 second per 2 target-language words across all phrases
    total_words = sum(len(p["target"].split()) for p in phrases)
    max_seconds = max(1, round(total_words / 2 * _S7_SECONDS_PER_TWO_WORDS))
    st.caption(
        f"Pass target: **≤ {max_seconds}s** "
        f"({total_words} target words · 1 sec per 2 words) "
        f"and **≥ {int(_S7_MIN_SIMILARITY*100)}%** accuracy."
    )

    # ── Record & Submit ──
    # No live timer: speed = actual audio recording length.
    st.markdown("#### 🎙️ Translate all phrases and record")
    audio = audio_input("s7")

    # Submit button appears only after the user records audio
    if audio and st.button("Submit & Score", type="primary",
                           use_container_width=True, key="s7_submit"):
        duration_s = max(1, round(_audio_duration_ms(audio) / 1000))
        full = ". ".join(p["target"] for p in phrases)
        r = do_score(session, audio, full, wh_lang, step=7, phrase_id=0)
        if r:
            st.session_state["s7_result"] = {
                "time":  duration_s,
                "score": r["score"],
            }

    if "s7_result" in st.session_state:
        res      = st.session_state["s7_result"]
        time_ok  = res["time"] <= max_seconds
        score_ok = res["score"] >= _S7_MIN_SIMILARITY
        passed   = time_ok and score_ok

        if passed:
            st.success(
                f"🎉 Passed! Translation speed: **{res['time']}s** · "
                f"**{int(res['score']*100)}%** accuracy."
            )
        else:
            problems = []
            if not time_ok:
                problems.append(
                    f"speed {res['time']}s > target {max_seconds}s"
                )
            if not score_ok:
                problems.append(
                    f"accuracy {int(res['score']*100)}% < "
                    f"{int(_S7_MIN_SIMILARITY*100)}% required"
                )
            st.warning(
                "Not quite there yet — " + " · ".join(problems) +
                ". You can still continue to Step 8."
            )

    # Character above phrases — shown once per step entry, not on button-click rerun
    if "s7_char_shown" not in st.session_state:
        st.session_state["s7_char_shown"] = True
        show_character("polyglot", "rare_bonus")

    # Phrase list in the middle (native shown, target hidden)
    phrase_table(phrases, show_native=True, show_target=False)

    # Navigation at the bottom — only after a recording was scored
    if "s7_result" in st.session_state:
        if st.button("Continue to Step 8 →", type="primary",
                     use_container_width=True, key="s7_continue"):
            st.session_state.pop("s7_result", None)
            st.session_state.pop("s7_char_shown", None)
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Lesson complete
# ═══════════════════════════════════════════════════════════════════════════
def _char_img_b64(character_key: str) -> str:
    """Повертає base64 PNG персонажа або порожній рядок."""
    from pathlib import Path as _Path
    import base64 as _b64
    p = ROOT / "assets" / "characters" / f"{character_key}.png"
    if p.exists():
        return _b64.b64encode(p.read_bytes()).decode()
    return ""


def render_complete(session: LessonSession):
    # ── Визначаємо персонажа і фразу ─────────────────────────────────────────
    from engine.characters import get_phrase as _get_phrase

    # Мова
    _nat_lang = session.state.native_lang or st.session_state.get("launcher_native", "English")

    # Прогрес streak — потрібен щоб вибрати правильну категорію фрази
    _streak = st.session_state.get("_cached_streak", 0)
    _category = "on_streak" if _streak > 1 else "on_lesson_complete"

    _char_data = _get_phrase("natalia", _category, lang=_nat_lang)
    _phrase    = _char_data["phrase"] if _char_data else "Great work! Keep it up 🎉"
    _char_name = _char_data["name"]   if _char_data else "Natalia"
    _img_b64   = _char_img_b64("natalia")
    _img_tag   = (
        '<img src="data:image/png;base64,' + _img_b64 + '" '
        'style="width:110px;height:110px;object-fit:cover;'
        'border-radius:50%;border:3px solid var(--mova-mint);'
        'box-shadow:0 4px 14px rgba(0,0,0,.15);margin-bottom:8px;" />'
        if _img_b64 else
        '<div style="font-size:4rem;margin-bottom:8px;">👩‍🏫</div>'
    )

    st.markdown(
        '<div class="cbanner" style="padding:32px 36px;">'
        '<div style="font-size:2.4rem;margin-bottom:6px;">🎉</div>'
        '<h2 style="color:var(--mova-ink);margin:0 0 20px 0;">Lesson Complete!</h2>'
        '<div style="display:flex;align-items:center;gap:22px;'
        'background:rgba(255,255,255,.45);border-radius:14px;'
        'padding:18px 22px;text-align:left;">'
        '<div style="flex-shrink:0;text-align:center;">'
        + _img_tag +
        '<div style="font-size:.75rem;font-weight:600;color:#E65100;margin-top:2px;">'
        + _char_name +
        '</div></div>'
        '<div style="font-size:1rem;color:#333;line-height:1.55;">'
        '&#128172; ' + _phrase +
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    # ── Зберігаємо прогрес (один раз) ────────────────────────────────────────
    if not st.session_state.get("_progress_saved"):
        session.complete()
        st.session_state["_progress_saved"] = True
        try:
            _llang = {"English":"en","Ukrainian":"uk","Spanish":"es","Korean":"ko"}.get(
                st.session_state.get("launcher_native","English"), "en")
            _lres  = on_lesson_complete(session.state.user_id, _llang)
            _ltoasts = [f"🎉 Урок завершено! +{_lres['xp_earned']} XP бонус"]
            if _lres.get("leveled_up"):
                _ltoasts.append(f"⭐ Новий рівень {_lres['level_num']}: {_lres['level_name']}!")
            for _bid, _bem, _bname, _bdesc in _lres.get("new_badges", []):
                _ltoasts.append(f"{_bem} Бейдж «{_bname}»: {_bdesc}!")
            streak = _lres.get("streak_current", 0)
            st.session_state["_cached_streak"] = streak
            if streak > 1:
                _ltoasts.append(f"🔥 Серія {streak} {'день' if streak == 1 else 'днів'}!")
            # Поліглот кожні 5 уроків — окремо під банером
            _lessons_done = st.session_state.get("_total_lessons_done", 0) + 1
            st.session_state["_total_lessons_done"] = _lessons_done
            st.session_state["_pending_toasts"] = (
                st.session_state.get("_pending_toasts", []) + _ltoasts
            )
        except Exception:
            pass

    module = _current_module()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Redo lesson", use_container_width=True, type="primary"):
            st.session_state.pop("_progress_saved", None)
            _clear_lesson()
            # Reset adaptive index so the sequence starts from step 1 again
            st.session_state.pop("_adaptive_lesson_id", None)
            st.session_state["lesson_step"] = 1
            st.rerun()
    with c2:
        # Custom mode has no fixed "next lesson" — show "Back to my phrases" instead
        if module == "custom":
            if st.button("📝 Back to my phrases", use_container_width=True):
                st.session_state.pop("_progress_saved", None)
                _clear_all()
                st.session_state["active_module"] = "custom"
                st.rerun()
        elif st.button("▶ Next lesson", use_container_width=True):
            # Load next lesson automatically — module-aware
            sess   = st.session_state["session"]
            state  = sess.state
            cfg    = _module_config(_current_module())
            try:
                df_all    = cfg["load"](str(cfg["db_path"]), state.native_lang, state.target_lang)
                lessons   = cfg["get_lessons"](df_all)
                next_id   = state.lesson_id + 1
                if next_id in lessons:
                    lang_pair = state.language_pair
                    lesson_df = cfg["get_lesson"](df_all, next_id)
                    st.session_state.pop("_progress_saved", None)
                    _clear_lesson()
                    # Clear adaptive so it reinitialises for the new lesson
                    st.session_state.pop("_adaptive_lesson_id", None)
                    st.session_state.update({
                        "session":     LessonSession(state.user_id, lesson_df, next_id,
                                                     state.native_lang, state.target_lang,
                                                     language_pair=lang_pair),
                        "lesson_step": 1,
                    })
                    st.rerun()
                else:
                    st.info("This was the last lesson!")
            except Exception as e:
                st.error(f"Error loading next lesson: {e}")
    with c3:
        if st.button("📚 Choose lesson", use_container_width=True):
            st.session_state.pop("_progress_saved", None)
            _clear_all()
            st.rerun()




# ═══════════════════════════════════════════════════════════════════════════
# Setup screen
# ═══════════════════════════════════════════════════════════════════════════
def render_setup():
    module = _current_module()
    cfg    = _module_config(module)

    st.markdown(f"""
    <div style="text-align:center;padding:48px 0 24px">
      <div style="font-size:3rem">{cfg['icon']}</div>
      <h1 style="color:var(--mova-ink);font-weight:600;margin:10px 0 4px">IMLLS — {cfg['label']}</h1>
      <p style="color:var(--mova-ink-3)">Intelligent Multilingual Language Learning System</p>
    </div>""", unsafe_allow_html=True)

    db_path = cfg["db_path"]
    if not db_path.exists():
        st.error(f"Database not found: `{db_path}`\n\nPlace the file in the `data/` folder.")
        st.stop()

    # ── Query-params bridge: wave clicked a lesson (all modules) ──────────────
    if cfg.get("load") is not None:
        _qp = st.query_params
        if "vnav_lesson" in _qp:
            try:
                from urllib.parse import unquote as _uq
                _qp_lid    = int(_qp["vnav_lesson"])
                _qp_native = _uq(_qp.get("vnav_native", "English"))
                _qp_target = _uq(_qp.get("vnav_target", "Ukrainian"))
                _qp_user   = _uq(_qp.get("vnav_user",   "student1"))
                if _qp_native not in LANGUAGES:
                    _qp_native = "English"
                if _qp_target not in LANGUAGES or _qp_target == _qp_native:
                    _qp_target = next(l for l in LANGUAGES if l != _qp_native)
                st.query_params.clear()
                # Vocab: lazy-load only the sheet that contains this lesson
                _qp_extra = {}
                if cfg.get("lang_suffix") == "vocab":
                    from engine.vocab_loader import get_topic_for_lesson
                    _qp_topic, _ = get_topic_for_lesson(str(db_path), _qp_lid)
                    if _qp_topic:
                        _qp_extra["topic"] = _qp_topic
                _qp_df        = cfg["load"](str(db_path), _qp_native, _qp_target,
                                            **_qp_extra)
                _qp_lesson_df = cfg["get_lesson"](_qp_df, _qp_lid)
                if not _qp_lesson_df.empty:
                    _qp_lp = (f"{WHISPER_LANG.get(_qp_native,'?')}"
                              f"-{WHISPER_LANG.get(_qp_target,'?')}-{cfg['lang_suffix']}")
                    st.session_state.update({
                        "session":     LessonSession(_qp_user, _qp_lesson_df,
                                                     _qp_lid, _qp_native, _qp_target,
                                                     language_pair=_qp_lp),
                        "lesson_step": 1,
                        "tts_lang":    TTS_LANG.get(_qp_target, "en"),
                        "wh_lang":     WHISPER_LANG.get(_qp_target),
                        "lang_pair":   _qp_lp,
                    })
                    st.rerun()
            except Exception:
                st.query_params.clear()

    # Pre-fill from the launcher if the user picked language / username there
    default_native = st.session_state.get("launcher_native", "English")
    if default_native not in LANGUAGES:
        default_native = "English"
    default_target = st.session_state.get("launcher_target", "Ukrainian")

    c1, c2 = st.columns(2)
    with c1:
        native = st.selectbox(
            "🌐 Native language", LANGUAGES,
            index=LANGUAGES.index(default_native),
        )
    with c2:
        target_options = [l for l in LANGUAGES if l != native]
        target_idx     = (target_options.index(default_target)
                          if default_target in target_options else 0)
        target = st.selectbox("🎯 Target language", target_options,
                              index=target_idx)

    # ── Load lessons ──────────────────────────────────────────────────────────
    # Vocab: lazy path — build lesson list from the lightweight index only.
    # No phrase text is read here; sheets are loaded on demand when the user
    # starts a lesson (_start_grammar_lesson passes topic= kwarg).
    if module == "vocab":
        from engine.vocab_loader import get_vocab_nav_data as _gnvd
        _nav_data = _gnvd(str(db_path))
        lessons       = sorted(l["gid"] for ls in _nav_data.values() for l in ls)
        df            = None  # not needed for vocab navigation
        counts_by_lid = {}    # phrase counts unknown until a sheet is loaded
    else:
        try:
            df = cfg["load"](str(db_path), native, target)
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()
        lessons       = cfg["get_lessons"](df)
        counts_by_lid = df.groupby("lesson_id").size().to_dict() if not df.empty else {}

    # Suffix language_pair with module so grammar/vocab progress are tracked separately
    lang_pair = f"{WHISPER_LANG.get(native,'?')}-{WHISPER_LANG.get(target,'?')}-{cfg['lang_suffix']}"
    default_user = st.session_state.get("launcher_user", "student1")
    user_id      = st.text_input("👤 Your name", value=default_user)

    if not lessons:
        st.warning(f"⚠️ No {cfg['label'].lower()} lessons available for {native} → {target}. "
                   "The Excel file might not have data for this language pair yet.")
        st.stop()

    # Auto-select lesson based on saved progress.
    # progress = {"last_completed_lesson": int, "last_step": int}
    # last_step convention: 1..8 = mid-lesson; 99 = lesson done
    progress      = get_progress(user_id, lang_pair) if user_id else None
    default_idx   = 0
    resume_step   = 1   # which step to start at when "Start" is pressed
    resume_msg    = None

    if progress:
        saved_lesson = progress["last_completed_lesson"]
        saved_step   = progress["last_step"]

        if saved_step >= 99:
            # Lesson fully done -> offer next lesson at step 1
            next_lesson = saved_lesson + 1
            if next_lesson in lessons:
                default_idx = lessons.index(next_lesson)
                resume_step = 1
                resume_msg  = f"▶ Continuing from {cfg['lesson_word']} {next_lesson} (last completed: {saved_lesson})"
            else:
                st.success(f"🎉 All {cfg['label'].lower()} lessons completed for this language pair!")
        else:
            # Mid-lesson -> resume at the same lesson + step
            if saved_lesson in lessons:
                default_idx = lessons.index(saved_lesson)
                resume_step = max(1, min(8, saved_step))
                resume_msg  = f"⏯ Resume {cfg['lesson_word']} {saved_lesson} at Step {resume_step}"

    # Build topics map: either from the workbook (grammar) or a topic loader (vocab)
    if cfg["topics"] == "from_df":
        # Grammar: optional topic_en / topic_uk columns in the workbook
        topics_map = get_grammar_topics(df, native_lang=native)
    elif cfg["topics"]:
        # Vocab: topic loader that takes the db path (index-only, fast)
        topics_map = cfg["topics"](str(db_path))
    else:
        topics_map = None

    # ── Vocabulary: hierarchical navigator ───────────────────────────────────
    if module == "vocab":
        _render_vocab_nav(
            df=df, cfg=cfg, db_path=db_path, lang_pair=lang_pair,
            native=native, target=target, user_id=user_id,
            lessons=lessons, default_idx=default_idx,
            resume_step=resume_step, resume_msg=resume_msg,
            progress=progress, counts_by_lid=counts_by_lid,
        )
        return  # nav handles st.rerun internally

    # ── Grammar / Reading: wave navigator ───────────────────────────────────
    _render_flat_wave_nav(
        df=df, cfg=cfg, lang_pair=lang_pair,
        native=native, target=target, user_id=user_id,
        lessons=lessons, default_idx=default_idx,
        resume_step=resume_step, resume_msg=resume_msg,
        progress=progress, counts_by_lid=counts_by_lid,
        topics_map=topics_map,
    )



# ═══════════════════════════════════════════════════════════════════════════
# Step 8 — Creative Generation (structure + semantic analysis)
# ═══════════════════════════════════════════════════════════════════════════
def step8(session: LessonSession, tts_lang, wh_lang):
    session.start_step(8)

    # Step 8 description depends on practice module
    module = _current_module()
    if module == "vocab":
        title = "Compose Your Own Examples"
        desc  = ("Create your own sentences using the new words and phrases "
                 "from this lesson. The system will check grammar.")
    else:
        title = "Grammar Check — Create Your Own Phrases"
        desc  = "Say or type phrases in the target language. The system will correct grammar errors."

    step_hdr(8, total=8, show_image=True)

    phrases = session.phrases()

    # Languages that have a GEC model on HuggingFace
    SUPPORTED_GEC_LANGS = {"en", "es", "ko"}
    gec_lang = wh_lang if wh_lang in SUPPORTED_GEC_LANGS else None

    # Крок 8 активний тільки для мов, для яких ми натренували модель
    if gec_lang is None:
        st.info("Grammar correction is available for English, Spanish and Korean only. "
                "This step is skipped for other languages.")
        if st.button("Continue →", type="primary", use_container_width=True):
            return True
        return False

    if not gec_available(gec_lang):
        st.warning(f"⚠️ GEC model for `{gec_lang}` could not be loaded from HuggingFace.")
        if st.button("Skip →", use_container_width=True):
            return True
        return False

    # Reference table
    with st.expander("📚 Lesson phrases for reference", expanded=False):
        phrase_table(phrases, show_native=True, show_target=True)

    st.markdown("---")
    n_phrases = st.slider("How many phrases to create?", 1, 5, 3, key="s8_n")

    input_mode = st.radio("Input method", ["🎙️ Voice", "⌨️ Text"],
                          horizontal=True, key="s8_mode")

    results = st.session_state.get("s8_results", [])

    # ── Voice input ───────────────────────────────────────────────────────
    if input_mode == "🎙️ Voice":
        st.markdown(f"Record yourself saying **{n_phrases} original phrase(s)**.")
        audio = audio_input("s8_voice")

        if st.button("Submit & Check Grammar", type="primary",
                     use_container_width=True, key="s8_submit_voice"):
            if not audio:
                st.warning("Please record audio first.")
            elif not whisper_available():
                st.warning("Whisper not installed.")
            else:
                with st.spinner("Transcribing…"):
                    import re
                    raw        = transcribe_bytes(audio, language=wh_lang)
                    candidates = [s.strip() for s in re.split(r"[.!?]", raw)
                                  if len(s.strip()) > 3][:n_phrases]

                if not candidates:
                    st.warning(f"Could not detect phrases. Transcribed: `{raw}`")
                else:
                    with st.spinner("Checking grammar…"):
                        results = [{"original": p, "corrected": gec_correct(p, gec_lang)}
                                   for p in candidates]
                    st.session_state["s8_results"] = results
                    session.score(raw, raw, step=8, phrase_id=0)

    # ── Text input ────────────────────────────────────────────────────────
    else:
        # Language-specific placeholder examples with typical errors
        PLACEHOLDERS = {
            "en": "It's a big room.\nShe have a red pen.\nThey was happy.",
            "es": "Yo tiene un perro.\nElla son alta.\nNosotros vamos a la escuela.",
            "ko": "나는 학교에 가요.\n그는 책을 읽어요.\n우리는 친구이에요.",
        }
        text_input = st.text_area(
            "Type your phrases (one per line):",
            placeholder=PLACEHOLDERS.get(gec_lang, ""),
            height=120, key="s8_text_input",
        )
        if st.button("Submit & Check Grammar", type="primary",
                     use_container_width=True, key="s8_submit_text"):
            lines = [l.strip() for l in text_input.strip().splitlines()
                     if l.strip()][:n_phrases]
            if not lines:
                st.warning("Please enter at least one phrase.")
            else:
                with st.spinner("Checking grammar…"):
                    results = [{"original": p, "corrected": gec_correct(p, gec_lang)}
                               for p in lines]
                st.session_state["s8_results"] = results
                session.score(text_input, text_input, step=8, phrase_id=0)

    # ── Results ───────────────────────────────────────────────────────────
    if results:
        st.markdown("---")
        st.markdown("### ✏️ Grammar Correction Results")

        any_corrected = False
        for item in results:
            orig  = item["original"]
            corr  = item["corrected"]
            changed = orig.strip().lower() != corr.strip().lower()
            if changed: any_corrected = True

            color  = "#c04060" if changed else "var(--mova-mint)"
            icon   = "✗" if changed else "✓"
            label  = "corrected" if changed else "correct"

            if changed:
                corr_html = (
                    "<div style='color:var(--mova-ink);font-size:1rem;margin-top:6px;'>"
                    "<span style='color:var(--mova-mint);'>GEC: </span>"
                    + corr + "</div>"
                )
            else:
                corr_html = ""

            html_block = (
                "<div style='background:var(--mova-card);border:1px solid var(--mova-line);"
                "border-radius:12px;padding:14px 20px;margin:8px 0;'>"
                f"<div style='color:#808090;font-size:.75rem;margin-bottom:6px;'>{icon} {label}</div>"
                f"<div style='color:var(--mova-ink);font-size:1rem;'>"
                f"<span style='color:#8888b8;'>You: </span>{orig}</div>"
                + corr_html + "</div>"
            )
            st.markdown(html_block, unsafe_allow_html=True)

        if not any_corrected:
            st.success("✓ All phrases are grammatically correct!")
            show_character("ai_bot", "on_lesson_complete")
        else:
            st.info("💡 Review the corrections above and practice the corrected versions.")
            corrections_count = sum(1 for r in st.session_state.get("s8_results", []) if r.get("changed"))
            show_character("ai_bot", "feedback", score=70, corrections=corrections_count)

    # ── Navigation ────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Try again", use_container_width=True, key="s8_retry"):
            st.session_state.pop("s8_results", None)
            st.rerun()
    with c2:
        if st.button("Complete Lesson ✓", type="primary",
                     use_container_width=True, key="s8_complete"):
            st.session_state.pop("s8_results", None)
            return True
    return False

# ═══════════════════════════════════════════════════════════════════════════
# State helpers
# ═══════════════════════════════════════════════════════════════════════════
def _clear_lesson():
    for k in list(st.session_state):
        if k.startswith(("s1_","s2_","s3_","s4_","s5_","s6_","s7_","s8_","up_")):
            del st.session_state[k]

def _clear_all():
    for k in list(st.session_state): del st.session_state[k]
    st.query_params.clear()


def _save_step_progress(sess: LessonSession, step: int):
    """Persist (lesson_id, step) so the user can resume right here next time.
    Saves at most once per (lesson_id, step) per session to avoid spamming the API."""
    key = (sess.state.lesson_id, step)
    if st.session_state.get("_last_saved_progress") == key:
        return
    try:
        save_progress(
            user_id               = sess.state.user_id,
            language_pair         = sess.state.language_pair,
            last_completed_lesson = sess.state.lesson_id,
            last_step             = int(step),
        )
        st.session_state["_last_saved_progress"] = key
    except Exception as e:
        print(f"[app] save_progress error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# Main
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

.step-header{background:var(--mova-card);border:1px solid var(--mova-line);border-radius:14px;padding:18px 26px;margin-bottom:18px;}
.step-num{font-family:'JetBrains Mono',monospace;color:var(--mova-indigo);font-size:.78rem;margin-bottom:4px;}
.step-title{color:var(--mova-ink);font-size:1.25rem;font-weight:600;}
.step-desc{color:var(--mova-ink-2);font-size:.88rem;margin-top:5px;}
.step-pills{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px;}
.pill{font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:3px 9px;border-radius:20px;background:var(--mova-card);color:var(--mova-ink-3);border:1px solid var(--mova-line);}
.pill-active{background:var(--mova-indigo-soft);color:var(--mova-indigo);border-color:var(--mova-indigo);}
.pill-done{background:var(--mova-mint-soft);color:var(--mova-mint);border-color:var(--mova-mint);}

.ptable{background:var(--mova-card);border:1px solid var(--mova-line-2);border-radius:12px;overflow:hidden;margin:10px 0;}
.prow{display:flex;align-items:center;padding:11px 18px;border-bottom:1px solid var(--mova-line);gap:14px;}
.prow:last-child{border-bottom:none;}
.prow:hover{background:var(--mova-surface-3);}
.pnum{font-family:'JetBrains Mono',monospace;color:var(--mova-indigo);font-size:.78rem;min-width:26px;}
.pnat{color:var(--mova-ink);flex:1;font-size:.98rem;}
.ptgt{color:var(--mova-ink);flex:1;font-size:.98rem;font-weight:500;}
.phide{color:var(--mova-ink-4);flex:1;font-style:italic;font-size:.85rem;}
.spass{background:var(--mova-mint-soft);color:var(--mova-mint);border-radius:5px;padding:2px 9px;font-size:.8rem;font-family:'JetBrains Mono',monospace;}
.sfail{background:var(--mova-coral-soft);color:var(--mova-coral-ink);border-radius:5px;padding:2px 9px;font-size:.8rem;font-family:'JetBrains Mono',monospace;}

/* Color-coded step type — left border tells the user what kind of step it is */
.step-header.step-type-reading     { border-left: 4px solid var(--mova-indigo); }
.step-header.step-type-listening   { border-left: 4px solid #8050b0; }
.step-header.step-type-matching    { border-left: 4px solid #40a0a0; }
.step-header.step-type-translation { border-left: 4px solid #40a060; }
.step-header.step-type-composing   { border-left: 4px solid #c06080; }

/* Big icon next to the step title */
.step-icon-row { display:flex; align-items:center; gap:16px; margin-top:6px; }
.step-icon-big { font-size:2.4rem; line-height:1; flex-shrink:0; }

/* Subtle pulse so users notice the 'record yourself' control */
@keyframes pulse-soft {
    0%, 100% { box-shadow: 0 0 0 0 rgba(96,96,208,.35); }
    50%      { box-shadow: 0 0 0 10px rgba(96,96,208,0); }
}
.stApp [data-testid="stAudioInput"] button {
    animation: pulse-soft 2.6s ease-in-out infinite;
    border-radius: 50%;
}

/* Step 3 — make choice buttons readable (dark theme instead of light) */
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

.timer{font-family:'JetBrains Mono',monospace;font-size:2.4rem;color:var(--mova-indigo);text-align:center;padding:14px;background:var(--mova-card);border-radius:12px;border:1px solid var(--mova-line);margin:10px 0;}
.prow-active{background:var(--mova-indigo-soft) !important;border-left:3px solid var(--mova-indigo);}
.prow-active .pnat{color:var(--mova-ink) !important;}
.prow-active .ptgt{color:var(--mova-ink) !important;font-weight:600;}
.pill-required{background:var(--mova-amber-soft);color:var(--mova-amber-ink);border-color:var(--mova-amber);}
.progress-bar-wrap{background:var(--mova-card);border-radius:8px;height:8px;margin:6px 0;overflow:hidden;}
.progress-bar-fill{height:8px;border-radius:8px;background:linear-gradient(90deg, var(--mova-indigo), #6E66FF);transition:width .4s;}
.progress-info{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--mova-ink-3);margin-bottom:2px;}
.cbanner{background:linear-gradient(135deg, var(--mova-mint-soft), var(--mova-indigo-soft));border:1px solid var(--mova-mint);border-radius:16px;padding:36px;text-align:center;}
audio{width:100%;border-radius:8px;margin:4px 0;}
</style>
""", unsafe_allow_html=True)


TOTAL_LESSONS = 173

STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6, 7: step7, 8: step8}
# Steps that cannot be skipped
REQUIRED_STEPS = {1, 3, 6, 7}


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive session initialiser
# ═══════════════════════════════════════════════════════════════════════════
def _init_adaptive_session(sess) -> None:
    """
    Determine and cache the adaptive step sequence for the current lesson.

    • Called once per lesson (keyed by lesson_id so it auto-resets on lesson change).
    • Reads the user's full log history (logs/{user_id}.csv) and trains the
      RandomForest when ≥ MIN_SAMPLES (25) interactions are available.
    • Stores in session_state:
        _adaptive_steps  – ordered list, e.g. [1,4,5,6,7,8] or [1,2,3,4,5,6,7,8]
        _adaptive_idx    – current position in that list (int, starts at 0)
        _adaptive_mode   – "cold_start" | "adaptive"
        _adaptive_lesson_id – lesson_id for which the above were computed
    """
    # Already initialised for this lesson?
    if (st.session_state.get("_adaptive_lesson_id") == sess.state.lesson_id
            and "_adaptive_steps" in st.session_state):
        return

    if not _ADAPTIVE_AVAILABLE:
        st.session_state.update({
            "_adaptive_steps":     _FULL_SEQ.copy(),
            "_adaptive_idx":       0,
            "_adaptive_mode":      "cold_start",
            "_adaptive_lesson_id": sess.state.lesson_id,
        })
        return

    engine = _AdaptiveEngine()

    # Read ALL historical interactions for this user (one CSV per user)
    try:
        log_rows = sess.logger.read_all()
    except Exception:
        log_rows = []

    engine.maybe_train(log_rows)

    # Compute per-phrase features if we have data for this lesson's phrases
    phrase_features: dict = {}
    if log_rows and engine.mode == "adaptive":
        try:
            phrases = sess.phrases()
            pid_set = {int(p.get("phrase_id", i)) for i, p in enumerate(phrases)}
            subset  = [r for r in log_rows
                       if int(r.get("phrase_id", -1)) in pid_set]
            if subset:
                sims  = [float(r["similarity"])       for r in subset]
                times = [float(r["response_time_ms"]) for r in subset]
                phrase_features = {
                    "avg_similarity":    sum(sims)  / len(sims),
                    "avg_response_time": sum(times) / len(times),
                    "total_attempts":    len(subset),
                }
        except Exception:
            pass

    steps = engine.get_steps(phrase_features)

    # If resuming mid-lesson, find the position of the current step in the sequence
    current_step = st.session_state.get("lesson_step", 1)
    try:
        init_idx = steps.index(current_step)
    except ValueError:
        # Current step not in adaptive sequence (e.g. resuming at step 2 with SKIP_EASY)
        # Fall back to the nearest prior step in the sequence
        prior = [i for i, s in enumerate(steps) if s <= current_step]
        init_idx = prior[-1] if prior else 0

    st.session_state.update({
        "_adaptive_steps":     steps,
        "_adaptive_idx":       init_idx,
        "_adaptive_mode":      engine.mode,
        "_adaptive_lesson_id": sess.state.lesson_id,
    })

def main(module: str = "grammar"):
    """Entry point. `module` may be "grammar" or "vocab"."""
    # Remember the chosen module for downstream functions
    st.session_state["practice_module"] = module
    cfg = _module_config(module)
    _inject_css()

    with st.sidebar:
        # ── Gamification widget (streak / XP / level) ────────────────────────
        _gami_user = (st.session_state["session"].state.user_id
                      if "session" in st.session_state else
                      st.session_state.get("launcher_user", "student1"))
        sidebar_widget(_gami_user)

        # ── Module navigator ─────────────────────────────────────────────────
        _SIDEBAR_MODULES = [
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
        for _mod_key, _mod_icon, _mod_name in _SIDEBAR_MODULES:
            _is_current = (module == _mod_key)
            if st.button(
                f"{_mod_icon} {_mod_name}",
                key=f"sb_mod_{_mod_key}",
                use_container_width=True,
                type="primary" if _is_current else "secondary",
                disabled=_is_current,
            ):
                _u = st.session_state.get("launcher_user", "student1")
                _n = st.session_state.get("launcher_native", "Ukrainian")
                _t = st.session_state.get("launcher_target", "English")
                for _k in list(st.session_state):
                    del st.session_state[_k]
                st.session_state["active_module"]   = _mod_key
                st.session_state["launcher_user"]   = _u
                st.session_state["launcher_native"] = _n
                st.session_state["launcher_target"] = _t
                st.query_params["module"] = _mod_key
                st.rerun()

        st.markdown("---")
        st.markdown(f"**{cfg['icon']} {cfg['label']}**")
        if "lesson_step" in st.session_state and "session" in st.session_state:
            sess  = st.session_state["session"]
            state = sess.state
            cur_step = st.session_state['lesson_step']
            try:
                if module == "custom":
                    # Custom: count this user's saved lessons for the pair
                    from engine.custom_store import list_user_lessons
                    cu_df = list_user_lessons(state.user_id,
                                              native_lang=state.native_lang,
                                              target_lang=state.target_lang)
                    total_lessons = max(len(cu_df), 1)
                else:
                    df_all_for_total = cfg["load"](str(cfg["db_path"]),
                                                   state.native_lang, state.target_lang)
                    total_lessons = max(len(cfg["get_lessons"](df_all_for_total)), 1)
            except Exception:
                total_lessons = TOTAL_LESSONS

            # ── User path: which exercise out of all, % to finish ──
            # "Completed" = fully-finished lessons (lesson_id - 1).
            # We still show "lesson N / M" so the user knows which one is open.
            completed_lessons = max(state.lesson_id - 1, 0)
            lesson_pct = round(completed_lessons / total_lessons * 100, 1)
            st.markdown(
                f'<div style="background:var(--mova-card);border:1px solid var(--mova-line);'
                f'border-radius:10px;padding:10px 12px;margin:4px 0 10px">'
                f'<div style="color:var(--mova-ink-2);font-size:.7rem;'
                f'font-family:\'JetBrains Mono\',monospace;'
                f'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">'
                f'Your path · {cfg["label"]}</div>'
                f'<div style="color:var(--mova-ink);font-size:1.05rem;font-weight:600">'
                f'{cfg["lesson_word"]} {state.lesson_id} / {total_lessons}'
                f'</div>'
                f'<div class="progress-info" style="margin-top:6px">'
                f'<span>{completed_lessons} done · {total_lessons - completed_lessons} to go</span>'
                f'<span>{lesson_pct}%</span></div>'
                f'<div class="progress-bar-wrap">'
                f'<div class="progress-bar-fill" style="width:{lesson_pct}%"></div></div>'
                f'</div>',
                unsafe_allow_html=True
            )
            _adp_steps = st.session_state.get("_adaptive_steps", list(range(1, 9)))
            _adp_idx   = st.session_state.get("_adaptive_idx", cur_step - 1)
            _adp_total = len(_adp_steps)
            _adp_pos   = _adp_idx + 1
            step_pct   = round(_adp_idx / max(_adp_total, 1) * 100, 0)
            _adp_mode  = st.session_state.get("_adaptive_mode", "cold_start")
            _adp_badge = (
                ' <span style="background:#1a3a1a;color:#34d0a0;font-size:.62rem;'
                'padding:1px 6px;border-radius:10px;font-family:\'JetBrains Mono\',monospace;'
                'vertical-align:middle">adaptive</span>'
                if _adp_mode == "adaptive" else ""
            )
            st.markdown(
                f'<div class="progress-info">'
                f'<span>Step {_adp_pos} / {_adp_total}{_adp_badge}</span>'
                f'<span>{"🔒" if cur_step in REQUIRED_STEPS else ""}</span></div>'
                f'<div class="progress-bar-wrap">'
                f'<div class="progress-bar-fill" style="width:{step_pct}%;background:linear-gradient(90deg, var(--mova-mint), #34D0A0)"></div></div>',
                unsafe_allow_html=True
            )
            st.caption(f"`{state.language_pair}`")

            # Step navigation — back / repeat
            st.markdown("---")
            st.caption("Step navigation")
            nav_c1, nav_c2 = st.columns(2)
            with nav_c1:
                _adp_idx_nav  = st.session_state.get("_adaptive_idx", 0)
                back_disabled = _adp_idx_nav <= 0
                if st.button("← Previous", disabled=back_disabled,
                             use_container_width=True, key="nav_back",
                             help="Go to the previous step"):
                    _clear_lesson()
                    _prev_idx = max(0, _adp_idx_nav - 1)
                    _prev_seq = st.session_state.get("_adaptive_steps", list(range(1, 9)))
                    st.session_state["_adaptive_idx"] = _prev_idx
                    st.session_state["lesson_step"]   = _prev_seq[_prev_idx]
                    st.rerun()
            with nav_c2:
                if st.button("🔄 Repeat", use_container_width=True,
                             key="nav_repeat",
                             help="Restart the current step"):
                    _clear_lesson()
                    # lesson_step stays the same, but per-step state is wiped
                    st.rerun()

            # Quick-jump dropdown — all 8 steps always visible;
            # steps skipped by adaptive are marked "(+ optional)"
            _adp_seq_nav  = st.session_state.get("_adaptive_steps", list(range(1, 9)))
            _adp_idx_jump = st.session_state.get("_adaptive_idx", 0)
            # Deduplicate adaptive seq (EXTRA_REPEAT may repeat step 2)
            _seen_j: set = set()
            _adp_unique = [s for s in _adp_seq_nav
                           if not (s in _seen_j or _seen_j.add(s))]  # type: ignore
            # Always offer all 8 steps; skipped ones go at end with "(+ optional)" label
            _all_8 = list(range(1, 9))
            _optional_steps = [s for s in _all_8 if s not in _adp_unique]
            _full_options = _adp_unique + _optional_steps

            def _step_label(s):
                base = f"Step {s}"
                if s in REQUIRED_STEPS:
                    base += " \U0001f512"
                if s in _optional_steps:
                    base += "  (+ optional)"
                return base

            jump_default = min(_adp_idx_jump, len(_full_options) - 1)
            jump_to = st.selectbox(
                "Jump to step",
                options=_full_options,
                index=jump_default,
                format_func=_step_label,
                key="nav_jump",
            )
            if jump_to != cur_step:
                if st.button(f"Go to Step {jump_to}",
                             use_container_width=True,
                             key="nav_go"):
                    _clear_lesson()
                    if jump_to in _adp_seq_nav:
                        # Step is in adaptive sequence -- use its index normally
                        _new_idx = _adp_seq_nav.index(jump_to)
                        st.session_state["_adaptive_idx"] = _new_idx
                    else:
                        # Optional step (skipped by adaptive) -- insert it after
                        # current position so adaptive flow continues after it
                        _cur_pos = st.session_state.get("_adaptive_idx", 0)
                        _new_seq = (
                            _adp_seq_nav[:_cur_pos] +
                            [jump_to] +
                            _adp_seq_nav[_cur_pos:]
                        )
                        st.session_state["_adaptive_steps"] = _new_seq
                        st.session_state["_adaptive_idx"]   = _cur_pos
                    st.session_state["lesson_step"] = jump_to
                    st.rerun()


            # -- Jump to lesson ----------------------------------------------------
            if module != "custom":
                st.markdown("---")
                st.caption("Jump to lesson")
                try:
                    _df_jmp = cfg["load"](str(cfg["db_path"]),
                                          state.native_lang, state.target_lang)
                    _all_lids = cfg["get_lessons"](_df_jmp)
                    _cur_idx  = (_all_lids.index(state.lesson_id)
                                 if state.lesson_id in _all_lids else 0)
                    _jump_lid = st.selectbox(
                        "lesson_jump_sel",
                        options=_all_lids,
                        index=_cur_idx,
                        format_func=lambda lid: f"{cfg['lesson_word']} {lid}",
                        key="sb_jump_lid",
                        label_visibility="collapsed",
                    )
                    if _jump_lid != state.lesson_id:
                        if st.button(
                            f"Go to {cfg['lesson_word']} {_jump_lid}",
                            use_container_width=True,
                            key="sb_go_lid",
                        ):
                            _ldf = cfg["get_lesson"](_df_jmp, _jump_lid)
                            _lp  = (f"{WHISPER_LANG.get(state.native_lang,'?')}"
                                    f"-{WHISPER_LANG.get(state.target_lang,'?')}"
                                    f"-{cfg['lang_suffix']}")
                            _clear_lesson()
                            st.session_state.pop("_progress_saved", None)
                            # Clear adaptive so it reinitialises for the new lesson
                            st.session_state.pop("_adaptive_lesson_id", None)
                            st.session_state.update({
                                "session":     LessonSession(
                                                   state.user_id, _ldf, _jump_lid,
                                                   state.native_lang, state.target_lang,
                                                   language_pair=_lp),
                                "lesson_step": 1,
                                "tts_lang":    TTS_LANG.get(state.target_lang, "en"),
                                "wh_lang":     WHISPER_LANG.get(state.target_lang),
                                "lang_pair":   _lp,
                            })
                            st.rerun()
                except Exception:
                    pass

            st.markdown("---")
        if st.button("\U0001f3e0 Main menu"):
            _clear_all()
            st.rerun()

    if "lesson_step" not in st.session_state:
        if module == "custom":
            # Custom mode owns its own setup screen (lesson list + add form)
            import custom_app
            custom_app.render_setup()
            return
        render_setup()
        return

    sess = st.session_state["session"]
    tts  = st.session_state["tts_lang"]
    wh   = st.session_state["wh_lang"]
    step = st.session_state["lesson_step"]

    if step > 8 or sess.state.lesson_complete:
        # Mark this lesson as fully completed (sentinel last_step=99)
        _save_step_progress(sess, 99)
        render_complete(sess)
        return

    # Auto-save progress on every step (deduped by (lesson_id, step))
    _save_step_progress(sess, step)

    # -- Initialise adaptive step sequence (once per lesson) ------
    _init_adaptive_session(sess)

    # -- Dispatch to current step -----------------------------------------
    done = STEPS[step](sess, tts, wh)

    if done:
        _clear_lesson()
        adp_seq = st.session_state.get("_adaptive_steps", list(range(1, 9)))
        adp_idx = st.session_state.get("_adaptive_idx", 0)
        nxt     = adp_idx + 1
        if nxt < len(adp_seq):
            st.session_state["_adaptive_idx"] = nxt
            st.session_state["lesson_step"]   = adp_seq[nxt]
        else:
            st.session_state["lesson_step"]   = 9   # triggers render_complete
        st.rerun()
