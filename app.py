"""
IMLLS - main launcher.

Run:
    streamlit run app.py

Lets the user choose between three practice modes at the start of a session:
  - Grammar  (uses grammar.py + data/imlls_database.xlsx)
  - Vocabulary (uses grammar.py with module="vocab" + data/vocabulary.xlsx)
  - Reading (uses reading_app.py + data/reading_lessons.xlsx)

Each module has its own progress tracked separately in the SessionLogger
via different language_pair suffixes.

The main menu also shows a per-module progress bar (% of all exercises
completed and "exercise N of M"), so the user always knows where they are
on their learning path.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# IMPORTANT: page config must be the very first Streamlit call
st.set_page_config(
    page_title="IMLLS",
    page_icon="🎓",
    layout="wide",
    # 'auto' lets the user keep the sidebar open inside lessons (for nav)
    # while still allowing them to collapse it on the launcher.
    initial_sidebar_state="auto",
)

# Import after set_page_config so the sub-apps' guarded set_page_config calls
# are no-ops (they're wrapped in try/except).
import grammar as grammar_app          # noqa: E402
import reading_app                  # noqa: E402
import custom_app                   # noqa: E402

# Used for fetching progress on the launcher
from engine.loader  import (                  # noqa: E402
    load_phrases, get_available_lessons, WHISPER_LANG,
)
from engine.vocab_loader import (             # noqa: E402
    load_vocab, get_available_vocab_lessons,
)
from engine.custom_store import list_user_lessons  # noqa: E402
from engine.logger import get_progress        # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Mova design system — Phase 1 integration
# ═══════════════════════════════════════════════════════════════════════════
MOVA_DIR = ROOT / "static" / "mova"

# Phase 1 CSS overrides on top of the three Mova files.
# - Relax container max-width so existing wide tables still fit (Mova default
#   is 720px which is too narrow for our two-column phrase tables).
# - Preserve a small no-op so future tweaks have a place to live.
_MOVA_LOCAL_OVERRIDES = """
:root { --mova-container-max: 1000px; }
"""


@st.cache_data(show_spinner=False)
def _read_mova_css() -> str:
    """Return the concatenated Mova CSS (tokens → overrides → components).
    Cached so the file system is hit only once per session.

    Sanitises the CSS so any literal `<style>` / `</style>` / `<\\/style>`
    sequences inside comments can't break out of our <style> block (this is
    a real risk — streamlit-overrides.css ships with a demo snippet in its
    own header comment that contains exactly these tokens).
    """
    parts = []
    for name in ("tokens.css", "streamlit-overrides.css", "components.css"):
        path = MOVA_DIR / name
        if path.exists():
            parts.append(f"/* ===== {name} ===== */\n{path.read_text(encoding='utf-8')}")
    parts.append(f"/* ===== local overrides ===== */{_MOVA_LOCAL_OVERRIDES}")
    css = "\n\n".join(parts)
    # Strip patterns that could prematurely close our <style> block when
    # Streamlit's markdown renderer pipes the string through markdown-it.
    css = (css.replace("</style>", "/* */")
              .replace("<\\/style>", "/* */")
              .replace("<style>", "/* */"))
    return css


def _inject_mova_css() -> None:
    """Inject the Mova design system: Google Fonts + 3 CSS files + local tweaks.

    Loaded on every page render. Sits on top of the sub-apps' existing CSS so
    Mova rules win via CSS cascade (and `!important` where present).

    Uses `@import` instead of <link> tags because Streamlit's markdown
    parser treats <link> as inline text in some versions.
    """
    if not MOVA_DIR.exists():
        return
    css = _read_mova_css()
    st.markdown(
        f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
{css}
</style>""",
        unsafe_allow_html=True,
    )


LANGUAGES = ["English", "Ukrainian", "Spanish", "Korean"]

DB_GRAMMAR  = ROOT / "data" / "imlls_database.xlsx"
DB_VOCAB    = ROOT / "data" / "vocabulary.xlsx"
DB_READING  = ROOT / "data" / "reading_lessons.xlsx"


MODULES = {
    "grammar": {
        "label":       "Grammar",
        "icon":        "🗣️",
        "tagline":     "Practice phrases - 8 steps with GEC correction",
        "color_from":  "var(--mova-card)",
        "color_to":    "var(--mova-card)",
    },
    "vocab": {
        "label":       "Vocabulary",
        "icon":        "📖",
        "tagline":     "Learn words by topic - Family, Food, Travel...",
        "color_from":  "var(--mova-card)",
        "color_to":    "var(--mova-card)",
    },
    "reading": {
        "label":       "Reading",
        "icon":        "🔤",
        "tagline":     "Learn to read with IPA audio",
        "color_from":  "var(--mova-card)",
        "color_to":    "var(--mova-card)",
    },
    "custom": {
        "label":       "My Phrases",
        "icon":        "📝",
        "tagline":     "Create your own lessons - same 8-step practice flow",
        "color_from":  "var(--mova-card)",
        "color_to":    "var(--mova-card)",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Progress helpers — used to show "Lesson N / M · X%" on every module card
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _count_grammar_lessons(native: str, target: str) -> int:
    """Total grammar lessons available for the chosen language pair."""
    try:
        df = load_phrases(str(DB_GRAMMAR), native, target)
        return len(get_available_lessons(df))
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def _count_vocab_lessons(native: str, target: str) -> int:
    """Total vocabulary lessons available (index only — no phrase text loaded)."""
    try:
        from engine.vocab_loader import get_vocab_nav_data
        nav_data = get_vocab_nav_data(str(DB_VOCAB))
        return sum(len(ls) for ls in nav_data.values())
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def _count_reading_lessons(lang: str = "en") -> int:
    """Total reading lessons available for the given language sheet."""
    try:
        import pandas as pd
        df = pd.read_excel(str(DB_READING), engine="openpyxl", sheet_name=lang)
        return int(df.iloc[:, 0].nunique())
    except Exception:
        return 0


def _module_progress(user_id: str, lang_pair: str, total: int) -> dict:
    """
    Look up the saved progress for `lang_pair` and convert it into:
      {"current": N, "total": M, "pct": float, "done": bool}
    """
    info = {"current": 1, "total": max(total, 0), "pct": 0.0, "done": False}
    if not user_id or total <= 0:
        return info
    try:
        p = get_progress(user_id, lang_pair)
    except Exception:
        p = None
    if not p:
        return info

    saved_lesson = int(p.get("last_completed_lesson") or 0)
    saved_step   = int(p.get("last_step") or 1)

    if saved_step >= 99:
        completed = min(saved_lesson, total)
        info.update({
            "current": min(completed + 1, total),
            "pct":     round(completed / total * 100, 1),
            "done":    completed >= total,
        })
    else:
        completed = max(saved_lesson - 1, 0)
        info.update({
            "current": min(saved_lesson, total),
            "pct":     round(completed / total * 100, 1),
        })
    return info


def _module_progress_card(module_key: str, native: str, target: str,
                          user_id: str) -> dict:
    """Returns the progress info to render on a module card."""
    if module_key == "reading":
        import streamlit as _st
        r_lang    = _st.session_state.get("r_lang", "en")
        total     = _count_reading_lessons(lang=r_lang)
        lang_pair = f"{r_lang}-reading"
        word      = "Lesson"
    elif module_key == "grammar":
        total     = _count_grammar_lessons(native, target)
        lang_pair = f"{WHISPER_LANG.get(native,'?')}-{WHISPER_LANG.get(target,'?')}-grammar"
        word      = "Lesson"
    elif module_key == "vocab":
        total     = _count_vocab_lessons(native, target)
        lang_pair = f"{WHISPER_LANG.get(native,'?')}-{WHISPER_LANG.get(target,'?')}-vocab"
        word      = "Topic"
    else:
        try:
            total = len(list_user_lessons(user_id, native_lang=native,
                                          target_lang=target))
        except Exception:
            total = 0
        lang_pair = f"{WHISPER_LANG.get(native,'?')}-{WHISPER_LANG.get(target,'?')}-custom"
        word      = "Lesson"
    pr = _module_progress(user_id, lang_pair, total)
    pr["lang_pair"]   = lang_pair
    pr["lesson_word"] = word
    pr["module_key"]  = module_key
    return pr


def render_launcher():
    st.markdown("""
    <style>
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
    .mode-card{
        background:var(--mova-card);
        border:1px solid var(--mova-line);
        border-radius:18px;
        padding:28px 22px 22px;
        text-align:center;
        transition:all .2s;
        height:100%;
    }
    .mode-card:hover{
        border-color:var(--mova-indigo);
        background:var(--mova-surface-3);
    }
    .mode-icon{font-size:3rem;margin-bottom:6px;}
    .mode-title{color:var(--mova-ink);font-size:1.4rem;font-weight:600;margin:6px 0;}
    .mode-tag{color:var(--mova-ink-2);font-size:.88rem;margin-bottom:14px;}
    .pb-wrap{background:var(--mova-surface-3);border-radius:8px;height:8px;margin:8px 0 6px;
             overflow:hidden;border:1px solid var(--mova-line);}
    .pb-fill{height:8px;border-radius:8px;
             background:linear-gradient(90deg, var(--mova-indigo), #6E66FF);transition:width .4s;}
    .pb-info{display:flex;justify-content:space-between;
             font-family:'JetBrains Mono',monospace;font-size:.72rem;
             color:#9090c0;margin-bottom:2px;}
    .pb-done{color:var(--mova-mint) !important;}
    .pb-fill-done{background:linear-gradient(90deg,var(--mova-mint),var(--mova-mint)) !important;}
    .pb-empty{color:var(--mova-ink-3) !important;}
    .menu-wrap{max-width:920px;margin:0 auto;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px">
      <div style="font-size:3.5rem">🎓</div>
      <h1 style="color:var(--mova-ink);font-weight:600;margin:10px 0 4px">IMLLS</h1>
      <p style="color:var(--mova-ink-3);font-size:1rem">
        Intelligent Multilingual Language Learning System
      </p>
      <p style="color:var(--mova-ink-3);margin-top:14px">
        Choose what you want to practice today:
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── User identity + language preferences (used for progress lookup) ──
    # Defaults persist across reruns via session_state.
    default_user   = st.session_state.get("launcher_user", "student1")
    default_native = st.session_state.get("launcher_native", "Ukrainian")
    default_target = st.session_state.get("launcher_target", "English")

    with st.container():
        c1, c2, c3 = st.columns([2, 1.3, 1.3])
        with c1:
            user_id = st.text_input("👤 Your name", value=default_user,
                                    key="launcher_user_input")
        with c2:
            native = st.selectbox("🌐 Native language", LANGUAGES,
                                  index=LANGUAGES.index(default_native)
                                  if default_native in LANGUAGES else 0,
                                  key="launcher_native_input")
        with c3:
            target_options = [l for l in LANGUAGES if l != native]
            target_default_idx = (target_options.index(default_target)
                                  if default_target in target_options else 0)
            target = st.selectbox("🎯 Target language", target_options,
                                  index=target_default_idx,
                                  key="launcher_target_input")

    # Persist for next render and for sub-apps to read
    st.session_state["launcher_user"]   = user_id
    st.session_state["launcher_native"] = native
    st.session_state["launcher_target"] = target

    st.markdown("<div style='margin:10px 0 18px'></div>", unsafe_allow_html=True)

    # One column per module — auto-extends if more modules are added later.
    cols = st.columns(len(MODULES))
    for col, (key, info) in zip(cols, MODULES.items()):
        with col:
            pr   = _module_progress_card(key, native, target, user_id)
            tot  = pr["total"]
            cur  = pr["current"]
            pct  = pr["pct"]
            word = pr["lesson_word"]

            # Build progress block — varies by state
            if tot <= 0:
                progress_html = (
                    '<div class="pb-info"><span class="pb-empty">No lessons yet</span>'
                    '<span class="pb-empty">—</span></div>'
                    '<div class="pb-wrap"><div class="pb-fill" style="width:0%"></div></div>'
                )
            elif pr["done"]:
                progress_html = (
                    f'<div class="pb-info">'
                    f'<span class="pb-done">All {tot} {word.lower()}s done!</span>'
                    f'<span class="pb-done">100%</span></div>'
                    f'<div class="pb-wrap">'
                    f'<div class="pb-fill pb-fill-done" style="width:100%"></div></div>'
                )
            else:
                progress_html = (
                    f'<div class="pb-info">'
                    f'<span>{word} {cur} / {tot}</span>'
                    f'<span>{pct:.0f}%</span></div>'
                    f'<div class="pb-wrap">'
                    f'<div class="pb-fill" style="width:{pct}%"></div></div>'
                )

            st.markdown(f"""
            <div class="mode-card">
              <div class="mode-icon">{info['icon']}</div>
              <div class="mode-title">{info['label']}</div>
              <div class="mode-tag">{info['tagline']}</div>
              {progress_html}
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Start {info['label']}", key=f"pick_{key}",
                         use_container_width=True, type="primary"):
                _switch_to(key)


def _switch_to(module_key: str):
    """Reset session state and remember the chosen module + user prefs."""
    # Save launcher prefs so sub-app can pre-fill them
    user   = st.session_state.get("launcher_user", "student1")
    native = st.session_state.get("launcher_native", "Ukrainian")
    target = st.session_state.get("launcher_target", "English")

    # Wipe everything so previous-module state can't bleed through
    for k in list(st.session_state):
        del st.session_state[k]
    st.session_state["active_module"]   = module_key
    st.session_state["launcher_user"]   = user
    st.session_state["launcher_native"] = native
    st.session_state["launcher_target"] = target
    st.query_params["module"] = module_key
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Main router
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # Mova design system — Phase 1. Loaded on every rerun so it overrides
    # the sub-apps' legacy CSS via cascade.
    _inject_mova_css()

    # Sub-apps may set this flag in their sidebar "Switch mode" button.
    if st.session_state.pop("_show_launcher", False):
        # Preserve launcher preferences across the reset
        user   = st.session_state.get("launcher_user", "student1")
        native = st.session_state.get("launcher_native", "Ukrainian")
        target = st.session_state.get("launcher_target", "English")
        for k in list(st.session_state):
            del st.session_state[k]
        st.session_state["launcher_user"]   = user
        st.session_state["launcher_native"] = native
        st.session_state["launcher_target"] = target
        st.query_params.clear()
        render_launcher()
        return

    # Restore module from URL query param (enables browser back/forward)
    qp_module = st.query_params.get("module")
    if qp_module and not st.session_state.get("active_module"):
        st.session_state["active_module"] = qp_module
    elif st.session_state.get("active_module") and not qp_module:
        # Session has module but URL doesn't — keep session as truth, update URL
        st.query_params["module"] = st.session_state["active_module"]

    active = st.session_state.get("active_module")

    if active is None:
        render_launcher()
        return

    if active == "grammar":
        grammar_app.main(module="grammar")
    elif active == "vocab":
        grammar_app.main(module="vocab")
    elif active == "reading":
        # Sync launcher target language → reading language code (only if not yet set)
        _TARGET_TO_RLANG = {
            "English":   "en",
            "Ukrainian": "uk",
            "Spanish":   "es",
            "Korean":    "ko",
        }
        _target = st.session_state.get("launcher_target", "English")
        if "r_lang" not in st.session_state:
            st.session_state["r_lang"] = _TARGET_TO_RLANG.get(_target, "en")
        reading_app.main()
    elif active == "custom":
        custom_app.main()
    else:
        # Unknown module - reset
        for k in list(st.session_state):
            del st.session_state[k]
        st.query_params.clear()
        render_launcher()


if __name__ == "__main__":
    main()
