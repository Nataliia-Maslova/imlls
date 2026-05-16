"""
IMLLS - main launcher.

Run:
    streamlit run main_app.py

Lets the user choose between three practice modes at the start of a session:
  - Grammar  (uses app.py + data/imlls_database.xlsx)
  - Vocabulary (uses app.py with module="vocab" + data/vocabulary.xlsx)
  - Reading (uses reading_app.py + data/reading_lessons.xlsx)

Each module has its own progress tracked separately in the SessionLogger
via different language_pair suffixes.
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
    initial_sidebar_state="collapsed",
)

# Import after set_page_config so the sub-apps' guarded set_page_config calls
# are no-ops (they're wrapped in try/except).
import grammar as grammar_app          # noqa: E402
import reading_app                  # noqa: E402


MODULES = {
    "grammar": {
        "label":       "Grammar",
        "icon":        "🗣️",
        "tagline":     "Practice phrases - 8 steps with GEC correction",
        "color_from":  "#16213e",
        "color_to":    "#1a1a2e",
    },
    "vocab": {
        "label":       "Vocabulary",
        "icon":        "📖",
        "tagline":     "Learn words by topic - Family, Food, Travel...",
        "color_from":  "#1a2e16",
        "color_to":    "#1a2e1a",
    },
    "reading": {
        "label":       "Reading",
        "icon":        "🔤",
        "tagline":     "English phonics - 80 lessons with IPA audio",
        "color_from":  "#2e1a16",
        "color_to":    "#2e1a1a",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Launcher screen
# ═══════════════════════════════════════════════════════════════════════════
def render_launcher():
    st.markdown("""
    <style>
    .stApp{background:#0d0d14;color:#e2e2f0;}
    #MainMenu,footer,header{visibility:hidden;}
    .mode-card{
        background:linear-gradient(135deg,#1a1a2e,#16213e);
        border:1px solid #2a2a4a;
        border-radius:18px;
        padding:32px 24px;
        text-align:center;
        transition:all .2s;
        cursor:pointer;
        height:100%;
    }
    .mode-card:hover{
        border-color:#5050b0;
        background:linear-gradient(135deg,#1e1e36,#1a253f);
    }
    .mode-icon{font-size:3rem;margin-bottom:10px;}
    .mode-title{color:#f0f0ff;font-size:1.4rem;font-weight:600;margin:8px 0;}
    .mode-tag{color:#8080a0;font-size:.9rem;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:48px 0 24px">
      <div style="font-size:3.5rem">🎓</div>
      <h1 style="color:#f0f0ff;font-weight:600;margin:10px 0 4px">IMLLS</h1>
      <p style="color:#606090;font-size:1rem">Intelligent Multilingual Language Learning System</p>
      <p style="color:#a0a0a0;margin-top:16px">Choose what you want to practice today:</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for col, (key, info) in zip(cols, MODULES.items()):
        with col:
            st.markdown(f"""
            <div class="mode-card">
              <div class="mode-icon">{info['icon']}</div>
              <div class="mode-title">{info['label']}</div>
              <div class="mode-tag">{info['tagline']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Start {info['label']}", key=f"pick_{key}",
                         use_container_width=True, type="primary"):
                _switch_to(key)


def _switch_to(module_key: str):
    """Reset session state and remember the chosen module."""
    # Wipe everything so previous-module state can't bleed through
    for k in list(st.session_state):
        del st.session_state[k]
    st.session_state["active_module"] = module_key
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Main router
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # Sub-apps may set this flag in their sidebar "Switch mode" button.
    if st.session_state.pop("_show_launcher", False):
        for k in list(st.session_state):
            del st.session_state[k]
        render_launcher()
        return

    active = st.session_state.get("active_module")

    if active is None:
        render_launcher()
        return

    if active == "grammar":
        grammar_app.main(module="grammar")
    elif active == "vocab":
        grammar_app.main(module="vocab")
    elif active == "reading":
        reading_app.main()
    else:
        # Unknown module - reset
        for k in list(st.session_state):
            del st.session_state[k]
        render_launcher()


if __name__ == "__main__":
    main()
