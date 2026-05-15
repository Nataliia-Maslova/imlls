"""
IMLLS — Intelligent Multilingual Language Learning System
Lesson-based flow: 7 steps, full lesson visible at once.
JS MediaRecorder for in-browser audio capture.

Run: streamlit run app.py
"""
import sys, base64, random, time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.loader  import load_phrases, get_lesson, get_available_lessons, TTS_LANG, WHISPER_LANG
from engine.session import LessonSession
from engine.scorer  import evaluate
from engine.tts     import get_audio_path
from engine.stt      import transcribe_bytes, whisper_available
from engine.logger  import SessionLogger, get_last_lesson, save_progress
from engine.gec import correct as gec_correct, gec_available

DB_PATH   = ROOT / "data" / "imlls_database.xlsx"
LANGUAGES = ["English", "Ukrainian", "Spanish", "Korean"]

st.set_page_config(page_title="IMLLS", page_icon="🗣️",
                   layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0d0d14;color:#e2e2f0;}
#MainMenu,footer,header{visibility:hidden;}

.step-header{background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #2a2a4a;border-radius:14px;padding:18px 26px;margin-bottom:18px;}
.step-num{font-family:'JetBrains Mono',monospace;color:#6060c0;font-size:.78rem;margin-bottom:4px;}
.step-title{color:#f0f0ff;font-size:1.25rem;font-weight:600;}
.step-desc{color:#8080a0;font-size:.88rem;margin-top:5px;}
.step-pills{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px;}
.pill{font-family:'JetBrains Mono',monospace;font-size:.7rem;padding:3px 9px;border-radius:20px;background:#1a1a2e;color:#5050a0;border:1px solid #2a2a3a;}
.pill-active{background:#2a2a5a;color:#a0a0ff;border-color:#5050b0;}
.pill-done{background:#0d2e1a;color:#40c070;border-color:#204030;}

.ptable{background:#13131e;border:1px solid #222236;border-radius:12px;overflow:hidden;margin:10px 0;}
.prow{display:flex;align-items:center;padding:11px 18px;border-bottom:1px solid #1e1e30;gap:14px;}
.prow:last-child{border-bottom:none;}
.prow:hover{background:#1a1a2a;}
.pnum{font-family:'JetBrains Mono',monospace;color:#4040a0;font-size:.73rem;min-width:26px;}
.pnat{color:#5a5a80;flex:1;font-size:.93rem;}
.ptgt{color:#9090b8;flex:1;font-size:.93rem;font-weight:500;}
.phide{color:#2a2a4a;flex:1;font-style:italic;font-size:.82rem;}
.spass{background:#0d2e1a;color:#40c070;border-radius:5px;padding:2px 9px;font-size:.8rem;font-family:'JetBrains Mono',monospace;}
.sfail{background:#2e0d0d;color:#c04040;border-radius:5px;padding:2px 9px;font-size:.8rem;font-family:'JetBrains Mono',monospace;}

.timer{font-family:'JetBrains Mono',monospace;font-size:2.4rem;color:#a0a0ff;text-align:center;padding:14px;background:#13131e;border-radius:12px;border:1px solid #2a2a4a;margin:10px 0;}
.prow-active{background:#1e1e40 !important;border-left:3px solid #6060d0;}
.prow-active .pnat{color:#c0c0ff !important;}
.prow-active .ptgt{color:#ffffff !important;font-weight:600;}
.pill-required{background:#2e1a0d;color:#d08040;border-color:#704020;}
.progress-bar-wrap{background:#1a1a2e;border-radius:8px;height:8px;margin:6px 0;overflow:hidden;}
.progress-bar-fill{height:8px;border-radius:8px;background:linear-gradient(90deg,#4040c0,#6060ff);transition:width .4s;}
.progress-info{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#5050a0;margin-bottom:2px;}
.cbanner{background:linear-gradient(135deg,#0d2e1a,#1a1a2e);border:1px solid #304030;border-radius:16px;padding:36px;text-align:center;}
audio{width:100%;border-radius:8px;margin:4px 0;}
</style>
""", unsafe_allow_html=True)


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
<div style="background:#13131e;border:1px solid #2a2a4a;border-radius:12px;padding:14px 18px;margin:8px 0;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <button id="pl-btn" onclick="plToggle()"
      style="background:#2a2a5a;color:#a0a0ff;border:1px solid #5050b0;border-radius:8px;
             padding:7px 18px;cursor:pointer;font-family:JetBrains Mono,monospace;font-size:.88rem;">
      ▶ Play All
    </button>
    <span id="pl-stat" style="color:#6060a0;font-size:.8rem;font-family:JetBrains Mono,monospace;">ready</span>
  </div>
  <div id="pl-bar" style="margin-top:10px;display:flex;gap:4px;flex-wrap:wrap;"></div>
</div>
<script>
(function(){{
  const srcs={srcs_js}, pauses={pauses_js}, n={n};
  let cur=-1, playing=false, aud=null, tmr=null;
  const bar=document.getElementById('pl-bar');
  for(let i=0;i<n;i++){{const d=document.createElement('div');d.id='dot-'+i;
    d.style.cssText='width:10px;height:10px;border-radius:50%;background:#2a2a5a;transition:.2s;';
    bar.appendChild(d);}}
  function dot(i,c){{const d=document.getElementById('dot-'+i);if(!d)return;
    d.style.background=c==='active'?'#a0a0ff':c==='done'?'#40c070':'#2a2a5a';}}
  function stop(){{if(aud){{aud.pause();aud=null;}}if(tmr){{clearTimeout(tmr);tmr=null;}}
    playing=false;cur=-1;document.getElementById('pl-btn').textContent='▶ Play All';
    document.getElementById('pl-btn').style.color='#a0a0ff';
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
    document.getElementById('pl-btn').style.color='#ff6060';playIdx(0);}}}};
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
  const srcs={srcs_js},pauses={pauses_js},n={n},uid='{uid}';
  let cur=-1,playing=false,aud=null,tmr=null;
  const bar=document.getElementById('pl-bar-'+uid);
  for(let i=0;i<n;i++){{const d=document.createElement('div');d.id='dot-'+uid+'-'+i;
    d.style.cssText='width:10px;height:10px;border-radius:50%;background:#2a2a5a;transition:.2s;';
    bar.appendChild(d);}}
  function dot(i,c){{const d=document.getElementById('dot-'+uid+'-'+i);if(!d)return;
    d.style.background=c==='active'?'#a0a0ff':c==='done'?'#40c070':'#2a2a5a';}}
  function stop(){{if(aud){{aud.pause();aud=null;}}if(tmr){{clearTimeout(tmr);tmr=null;}}
    playing=false;cur=-1;
    document.getElementById('pl-btn-'+uid).textContent='▶ Play All';
    document.getElementById('pl-btn-'+uid).style.color='#a0a0ff';
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
          document.getElementById('pl-btn-'+uid).style.color='#ff6060';playIdx(0);}}}};
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
            row_cls = ' class="prow" style="background:#1e1e35"'
        else:
            row_cls = ' class="prow"'
        html += f'<div{row_cls}>{num}{nat}{tgt}{sc}</div>'
    st.markdown(f'<div class="ptable">{html}</div>', unsafe_allow_html=True)


def step_hdr(step, title, desc, total=8):
    pills = "".join(
        f'<span class="pill {"pill-active" if s==step else "pill-done" if s<step else "pill-required" if s in REQUIRED_STEPS and s>step else ""}">'
        f'{"🔒" if s in REQUIRED_STEPS and s > step else s}</span>'
        for s in range(1, total+1)
    )
    req_note = ' <span style="color:#d08040;font-size:.72rem;">🔒 required</span>' if step in REQUIRED_STEPS else ''
    st.markdown(f"""
    <div class="step-header">
      <div class="step-pills">{pills}</div>
      <div class="step-num">STEP {step} / {total}{req_note}</div>
      <div class="step-title">{title}</div>
      <div class="step-desc">{desc}</div>
    </div>""", unsafe_allow_html=True)


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
    step_hdr(1, "Read All Phrases",
             "Read every phrase in the target language. Both languages are visible.")
    phrases = session.phrases()
    scores  = st.session_state.get("s1_scores", {})
    phrase_table(phrases, show_native=True, show_target=True, scores=scores)

    st.markdown("#### 🎙️ Record yourself reading all phrases")
    audio = audio_input("s1")

    c1, c2 = st.columns([3,1])
    with c1:
        if st.button("Submit & Check", type="primary", use_container_width=True, key="s1_submit"):
            if not audio:
                st.warning("Please record or upload audio first.")
            else:
                full = ". ".join(p["target"] for p in phrases)
                r    = do_score(session, audio, full, wh_lang, step=1, phrase_id=0)
                if r:
                    scores = {i: r for i in range(len(phrases))}
                    st.session_state["s1_scores"] = scores
                    st.markdown(f"{'✓' if r['passed'] else '✗'} **{int(r['score']*100)}%** — `{r['transcribed']}`")
    with c2:
        if st.button("Next →", use_container_width=True, key="s1_next"):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Listen & Repeat (auto-playlist with pause, both languages visible)
# ═══════════════════════════════════════════════════════════════════════════
def step2(session: LessonSession, tts_lang, wh_lang):
    session.start_step(2)
    step_hdr(2, "Listen & Repeat",
             "Listen to each phrase and repeat it aloud. Phrases play automatically with pauses.")
    phrases   = session.phrases()
    active_idx = st.session_state.get("s2_active", None)
    phrase_table(phrases, show_native=True, show_target=True, active_idx=active_idx)

    paths  = [get_audio_path(p["target"], tts_lang) for p in phrases]
    pauses = [phrase_pause(p["target"]) for p in phrases]

    st.markdown("Press **▶ Play All** to start. Repeat each phrase aloud during the pause.")
    playlist_html = autoplaylist_html_with_highlight(paths, pauses, uid="s2")
    components.html(playlist_html, height=120, scrolling=False)

    if st.button("Continue to Step 3 →", type="primary", use_container_width=True):
        st.session_state.pop("s2_active", None)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Listen & Match (hear target → pick correct native from shuffled list)
# ═══════════════════════════════════════════════════════════════════════════
def step3(session: LessonSession, tts_lang, wh_lang):
    session.start_step(3)
    step_hdr(3, "Listen & Match",
             "Listen to the phrase and find its translation from the shuffled list.")
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
            f'<span style="color:{"#2a7a4a" if v else "#7a2a2a"};flex:1">'
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
        for choice in st.session_state[f"s3_opts_{idx}"]:
            if st.button(choice, key=f"s3_{idx}_{choice[:20]}", use_container_width=True):
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
                if correct: st.success("✓ Correct!")
                else: st.error(f"✗ Wrong — answer: **{p['native']}**")
                time.sleep(0.4)
                st.rerun()

    if idx >= len(phrases):
        ok = sum(1 for v in scores.values() if v)
        st.success(f"Done! {ok}/{len(phrases)} correct.")
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
    step_hdr(4, "Speed Reading",
             "Press Start Timer, then read all phrases as fast as you can and record!")
    phrases = session.phrases()
    phrase_table(phrases, show_native=False, show_target=True)

    # ── Timer ──
    if "s4_start" in st.session_state:
        elapsed = int(time.time() - st.session_state["s4_start"])
        m, s = divmod(elapsed, 60)
        color = "#ff6060" if elapsed > 60 else "#a0a0ff"
        st.markdown(f'<div class="timer" style="color:{color}">{m:02d}:{s:02d}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="timer" style="color:#3a3a6a">00:00</div>', unsafe_allow_html=True)
        if st.button("▶ Start Timer", type="primary", use_container_width=True, key="s4_timer"):
            st.session_state["s4_start"] = time.time()
            st.rerun()
        return False

    # ── Record & Submit ──
    st.markdown("#### 🎙️ Record yourself reading all phrases")
    audio = audio_input("s4")

    if st.button("Submit & Score", type="primary", use_container_width=True, key="s4_submit"):
        if not audio:
            st.warning("Please record or upload audio first.")
        else:
            total_time = int(time.time() - st.session_state["s4_start"])
            full = ". ".join(p["target"] for p in phrases)
            r = do_score(session, audio, full, wh_lang, step=4, phrase_id=0)
            if r:
                st.session_state["s4_result"] = {"time": total_time, "score": r["score"], "text": r["transcribed"]}

    if "s4_result" in st.session_state:
        res = st.session_state["s4_result"]
        st.success(f"🏁 {res['time']}s — {int(res['score']*100)}% match")
        st.caption(f"Transcribed: {res['text']}")

    c1, c2 = st.columns([3,1])
    with c2:
        if st.button("Done →", use_container_width=True, key="s4_done"):
            for k in ["s4_start","s4_result"]: st.session_state.pop(k, None)
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Shadowing (auto-playlist, see native only, repeat target aloud)
# ═══════════════════════════════════════════════════════════════════════════
def step5(session: LessonSession, tts_lang, wh_lang):
    session.start_step(5)
    step_hdr(5, "Shadowing",
             "Listen to each phrase and repeat it in the target language. Only native translation shown.")
    phrases = session.phrases()
    active_idx = st.session_state.get("s5_active", None)
    phrase_table(phrases, show_native=True, show_target=False, active_idx=active_idx)

    paths  = [get_audio_path(p["target"], tts_lang) for p in phrases]
    pauses = [phrase_pause(p["target"]) for p in phrases]

    st.markdown("Press **▶ Play All** — listen and repeat each phrase during the pause.")
    playlist_html = autoplaylist_html_with_highlight(paths, pauses, uid="s5")
    components.html(playlist_html, height=120, scrolling=False)

    if st.button("Continue to Step 6 →", type="primary", use_container_width=True):
        st.session_state.pop("s5_active", None)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Active Translation (all phrases at once, single recording)
# ═══════════════════════════════════════════════════════════════════════════
def step6(session: LessonSession, tts_lang, wh_lang):
    session.start_step(6)
    step_hdr(6, "Active Translation",
             "Translate all phrases aloud in one go. Only native language shown.")
    phrases = session.phrases()
    scores  = st.session_state.get("s6_scores", {})
    phrase_table(phrases, show_native=True, show_target=False, scores=scores)

    st.markdown("#### 🎙️ Record yourself translating all phrases")
    audio = audio_input("s6")

    c1, c2 = st.columns([3,1])
    with c1:
        if audio and st.button("Submit & Score", type="primary", use_container_width=True, key="s6_submit"):
            full = ". ".join(p["target"] for p in phrases)
            r = do_score(session, audio, full, wh_lang, step=6, phrase_id=0)
            if r:
                scores = {i: r for i in range(len(phrases))}
                st.session_state["s6_scores"] = scores
                st.session_state["s6_done"] = True
                st.markdown(f"{'✓' if r['passed'] else '✗'} **{int(r['score']*100)}%** — `{r['transcribed']}`")
    with c2:
        if st.button("Next →", use_container_width=True, key="s6_next"):
            st.session_state["s6_idx"] = 0
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — Speed Translation (native only, timer, translate all)
# ═══════════════════════════════════════════════════════════════════════════
def step7(session: LessonSession, tts_lang, wh_lang):
    session.start_step(7)
    step_hdr(7, "Speed Translation",
             "Press Start Timer, then translate all phrases as fast as possible!")
    phrases = session.phrases()
    phrase_table(phrases, show_native=True, show_target=False)

    # ── Timer ──
    if "s7_start" in st.session_state:
        elapsed = int(time.time() - st.session_state["s7_start"])
        m, s  = divmod(elapsed, 60)
        color = "#ff6060" if elapsed > 60 else "#a0a0ff"
        st.markdown(f'<div class="timer" style="color:{color}">{m:02d}:{s:02d}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="timer" style="color:#3a3a6a">00:00</div>', unsafe_allow_html=True)
        if st.button("▶ Start Timer", type="primary", use_container_width=True, key="s7_timer"):
            st.session_state["s7_start"] = time.time()
            st.rerun()
        return False

    # ── Record & Submit ──
    st.markdown("#### 🎙️ Translate all phrases and record")
    audio = audio_input("s7")

    if st.button("Submit & Finish!", type="primary", use_container_width=True, key="s7_submit"):
        if not audio:
            st.warning("Please record or upload audio first.")
        else:
            total_time = int(time.time() - st.session_state["s7_start"])
            full = ". ".join(p["target"] for p in phrases)
            r = do_score(session, audio, full, wh_lang, step=7, phrase_id=0)
            if r:
                st.session_state["s7_result"] = {"time": total_time, "score": r["score"]}

    if "s7_result" in st.session_state:
        res = st.session_state["s7_result"]
        st.success(f"🏁 Done in **{res['time']}s** with **{int(res['score']*100)}%** accuracy!")
        if st.button("Complete Lesson ✓", type="primary", key="s7_complete"):
            for k in ["s7_start","s7_result"]: st.session_state.pop(k, None)
            return True

    if st.button("Skip →", use_container_width=True, key="s7_skip"):
        for k in ["s7_start","s7_result"]: st.session_state.pop(k, None)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Lesson complete
# ═══════════════════════════════════════════════════════════════════════════
def render_complete(session: LessonSession):
    logs   = session.logger.read_all()
    total  = len(logs)
    passed = sum(1 for r in logs if str(r.get("success")) == "1")
    rate   = passed / total * 100 if total else 0

    st.markdown(f"""
    <div class="cbanner">
      <div style="font-size:2.8rem">🎉</div>
      <h2 style="color:#f0f0ff;margin:10px 0">Lesson Complete!</h2>
      <p style="color:#a0a0c0">
        Pass rate: <strong style="color:#40c070">{rate:.0f}%</strong>
        ({passed}/{total} checks passed)
      </p>
    </div>""", unsafe_allow_html=True)

    # Save progress when lesson is completed
    if not st.session_state.get("_progress_saved"):
        session.complete()
        st.session_state["_progress_saved"] = True

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Redo lesson", use_container_width=True, type="primary"):
            st.session_state.pop("_progress_saved", None)
            _clear_lesson()
            st.session_state["lesson_step"] = 1
            st.rerun()
    with c2:
        if st.button("▶ Next lesson", use_container_width=True):
            # Load next lesson automatically
            sess   = st.session_state["session"]
            state  = sess.state
            try:
                df_all    = load_phrases(str(DB_PATH), state.native_lang, state.target_lang)
                lessons   = get_available_lessons(df_all)
                next_id   = state.lesson_id + 1
                if next_id in lessons:
                    lang_pair = state.language_pair
                    lesson_df = get_lesson(df_all, next_id)
                    st.session_state.pop("_progress_saved", None)
                    _clear_lesson()
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
    st.markdown("""
    <div style="text-align:center;padding:48px 0 24px">
      <div style="font-size:3rem">🗣️</div>
      <h1 style="color:#f0f0ff;font-weight:600;margin:10px 0 4px">IMLLS</h1>
      <p style="color:#606090">Intelligent Multilingual Language Learning System</p>
    </div>""", unsafe_allow_html=True)

    if not DB_PATH.exists():
        st.error(f"Database not found: `{DB_PATH}`\n\nPlace `imlls_database.xlsx` in the `data/` folder.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1: native = st.selectbox("🌐 Native language", LANGUAGES)
    with c2: target = st.selectbox("🎯 Target language", [l for l in LANGUAGES if l != native])

    try:
        df = load_phrases(str(DB_PATH), native, target)
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()

    lessons      = get_available_lessons(df)
    lang_pair    = f"{WHISPER_LANG.get(native,'?')}-{WHISPER_LANG.get(target,'?')}"
    user_id      = st.text_input("👤 Your name", value="student1")

    # Auto-select next lesson based on saved progress
    last_done    = get_last_lesson(user_id, lang_pair) if user_id else None
    default_idx  = 0
    if last_done is not None:
        next_lesson = last_done + 1
        if next_lesson in lessons:
            default_idx = lessons.index(next_lesson)
            st.info(f"▶ Continuing from Lesson {next_lesson} (last completed: {last_done})")
        else:
            st.success("🎉 All lessons completed for this language pair!")

    lesson_id = st.selectbox("📚 Lesson", lessons,
                              index=default_idx,
                              format_func=lambda x: f"Lesson {x}")
    lesson_df = get_lesson(df, lesson_id)
    st.caption(f"**{len(lesson_df)} phrases** · pair: `{lang_pair}`")

    if st.button("▶ Start Lesson", type="primary", use_container_width=True):
        st.session_state.update({
            "session":      LessonSession(user_id, lesson_df, lesson_id,
                                          native, target,
                                          language_pair=lang_pair),
            "lesson_step":  1,
            "tts_lang":     TTS_LANG.get(target, "en"),
            "wh_lang":      WHISPER_LANG.get(target),
            "lang_pair":    lang_pair,
        })
        st.rerun()



# ═══════════════════════════════════════════════════════════════════════════
# Step 8 — Creative Generation (structure + semantic analysis)
# ═══════════════════════════════════════════════════════════════════════════
def step8(session: LessonSession, tts_lang, wh_lang):
    session.start_step(8)
    step_hdr(8, "Grammar Check — Create Your Own Phrases",
             "Say or type phrases in the target language. The system will correct grammar errors.",
             total=8)

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

            color  = "#c04060" if changed else "#40c070"
            icon   = "✗" if changed else "✓"
            label  = "corrected" if changed else "correct"

            if changed:
                corr_html = (
                    "<div style='color:#e0e0ff;font-size:1rem;margin-top:6px;'>"
                    "<span style='color:#40c070;'>GEC: </span>"
                    + corr + "</div>"
                )
            else:
                corr_html = ""

            html_block = (
                "<div style='background:#13131e;border:1px solid #2a2a4a;"
                "border-radius:12px;padding:14px 20px;margin:8px 0;'>"
                f"<div style='color:#808090;font-size:.75rem;margin-bottom:6px;'>{icon} {label}</div>"
                f"<div style='color:#e0e0ff;font-size:1rem;'>"
                f"<span style='color:#8888b8;'>You: </span>{orig}</div>"
                + corr_html + "</div>"
            )
            st.markdown(html_block, unsafe_allow_html=True)

        if not any_corrected:
            st.success("✓ All phrases are grammatically correct!")
        else:
            st.info("💡 Review the corrections above and practice the corrected versions.")

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
        if k.startswith(("s1_","s2_","s3_","s4_","s5_","s6_","s7_","up_")):
            del st.session_state[k]

def _clear_all():
    for k in list(st.session_state): del st.session_state[k]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
TOTAL_LESSONS = 173

STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6, 7: step7, 8: step8}
# Steps that cannot be skipped
REQUIRED_STEPS = {1, 3, 6, 7}

def main():
    with st.sidebar:
        if "lesson_step" in st.session_state and "session" in st.session_state:
            sess  = st.session_state["session"]
            state = sess.state
            cur_step = st.session_state['lesson_step']
            # Global progress
            lesson_pct = round((state.lesson_id - 1) / TOTAL_LESSONS * 100, 1)
            st.markdown(
                f'<div class="progress-info">'
                f'<span>Lesson {state.lesson_id} / {TOTAL_LESSONS}</span>'
                f'<span>{lesson_pct}%</span></div>'
                f'<div class="progress-bar-wrap">'
                f'<div class="progress-bar-fill" style="width:{lesson_pct}%"></div></div>',
                unsafe_allow_html=True
            )
            # Step progress
            step_pct = round((cur_step - 1) / 8 * 100, 0)
            st.markdown(
                f'<div class="progress-info">'
                f'<span>Step {cur_step} / 8</span>'
                f'<span>{"🔒" if cur_step in REQUIRED_STEPS else ""}</span></div>'
                f'<div class="progress-bar-wrap">'
                f'<div class="progress-bar-fill" style="width:{step_pct}%;background:linear-gradient(90deg,#205040,#40c070)"></div></div>',
                unsafe_allow_html=True
            )
            st.caption(f"`{state.language_pair}`")
            logs = sess.logger.read_all()
            if logs:
                p = sum(1 for r in logs if str(r.get("success"))=="1")
                st.metric("Checks", len(logs))
                st.metric("Pass rate", f"{p/len(logs)*100:.0f}%")
            st.markdown("---")
        if st.button("🏠 Main menu"): _clear_all(); st.rerun()

    if "lesson_step" not in st.session_state:
        render_setup(); return

    sess = st.session_state["session"]
    tts  = st.session_state["tts_lang"]
    wh   = st.session_state["wh_lang"]
    step = st.session_state["lesson_step"]

    if step > 8 or sess.state.lesson_complete:
        render_complete(sess); return

    fn = STEPS.get(step)
    if fn:
        done = fn(sess, tts, wh)
        if done:
            _clear_lesson()
            st.session_state["lesson_step"] = step + 1
            st.rerun()
        # Show skip button only for optional steps
        elif step not in REQUIRED_STEPS:
            st.markdown("---")
            if st.button("⏭ Skip this step", key=f"skip_{step}",
                         help="This step is optional — you can skip it"):
                _clear_lesson()
                st.session_state["lesson_step"] = step + 1
                st.rerun()


if __name__ == "__main__":
    main()
