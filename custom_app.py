"""
custom_app.py — "My phrases" practice mode.

Lets the user create their own lessons (lesson name + native↔target pairs)
and then run them through the exact same 8-step flow as grammar / vocabulary.
Storage: data/custom_phrases.csv (mirrored to Google Sheets if configured).

Run via app.py launcher; direct entry: ``custom_app.main()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine.custom_store import (
    add_lesson, delete_lesson, get_lesson_phrases,
    list_user_lessons, parse_pairs_text, rename_lesson,
)
from engine.loader  import TTS_LANG, WHISPER_LANG
from engine.session import LessonSession

import grammar as grammar_app  # we reuse its 8-step machinery


LANGUAGES = ["English", "Ukrainian", "Spanish", "Korean"]


# ─── Styling reused from grammar module ───────────────────────────────────

def _inject_css():
    grammar_app._inject_css()


# ─── Setup screen ─────────────────────────────────────────────────────────

def render_setup():
    _inject_css()

    # ── Sidebar: always visible on setup screen ───────────────────────────────
    _MODS = [
        ("grammar", "🗣️", "Grammar"),
        ("vocab",   "📖", "Vocabulary"),
        ("reading", "🔤", "Reading"),
        ("custom",  "📝", "My Phrases"),
    ]
    with st.sidebar:
        st.markdown(
            '<div style="font-size:.7rem;color:var(--mova-ink-3);'
            'text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">'
            'Module</div>',
            unsafe_allow_html=True,
        )
        for _mk, _mi, _mn in _MODS:
            _active = (_mk == "custom")
            if st.button(
                f"{_mi} {_mn}",
                key=f"cu_sb_{_mk}",
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
        if st.button("🏠 Main menu", key="cu_setup_home"):
            for _k in list(st.session_state):
                del st.session_state[_k]
            st.query_params.clear()
            st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:36px 0 18px">
      <div style="font-size:3rem">📝</div>
      <h1 style="color:var(--mova-ink);font-weight:600;margin:8px 0 4px">My Phrases</h1>
      <p style="color:var(--mova-ink-3)">
        Створи свій урок із власних фраз — і тренуй той самий 8-крокевий цикл.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Identity + language pair ──────────────────────────────────────────
    default_user   = st.session_state.get("launcher_user",   "student1")
    default_native = st.session_state.get("launcher_native", "Ukrainian")
    default_target = st.session_state.get("launcher_target", "English")

    cA, cB, cC = st.columns([2, 1.3, 1.3])
    with cA:
        user_id = st.text_input("👤 Your name", value=default_user, key="cu_user")
    with cB:
        if default_native not in LANGUAGES:
            default_native = "Ukrainian"
        native = st.selectbox("🌐 Native", LANGUAGES,
                              index=LANGUAGES.index(default_native), key="cu_native")
    with cC:
        target_opts = [l for l in LANGUAGES if l != native]
        tdef = (target_opts.index(default_target)
                if default_target in target_opts else 0)
        target = st.selectbox("🎯 Target", target_opts, index=tdef, key="cu_target")

    lang_pair = f"{WHISPER_LANG.get(native,'?')}-{WHISPER_LANG.get(target,'?')}-custom"

    # ── Existing lessons for this user + pair ─────────────────────────────
    lessons_df = list_user_lessons(user_id, native_lang=native, target_lang=target)

    st.markdown("---")
    st.markdown(f"### 📚 Ваші уроки ({native} → {target})")
    if lessons_df.empty:
        st.info("Поки що жодного уроку для цієї пари. Створіть перший нижче ↓")
    else:
        for _, row in lessons_df.iterrows():
            lid = int(row["lesson_id"])
            with st.container():
                c1, c2, c3, c4 = st.columns([5, 1.2, 1.2, 1.2])
                with c1:
                    st.markdown(
                        f"**{row['lesson_name']}** "
                        f"<span style='color:var(--mova-ink-3);font-size:.8rem'>· {row['phrases']} phrases · id {lid}</span>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("▶ Start", key=f"cu_start_{lid}",
                                 type="primary", use_container_width=True):
                        _start_lesson(user_id, lid, native, target, lang_pair)
                with c3:
                    if st.button("✏ Edit name", key=f"cu_edit_{lid}",
                                 use_container_width=True):
                        st.session_state[f"cu_edit_open_{lid}"] = True
                with c4:
                    if st.button("🗑 Delete", key=f"cu_del_{lid}",
                                 use_container_width=True):
                        st.session_state[f"cu_del_confirm_{lid}"] = True

                if st.session_state.get(f"cu_edit_open_{lid}"):
                    new_name = st.text_input(
                        "Нова назва уроку",
                        value=row["lesson_name"],
                        key=f"cu_new_name_{lid}",
                    )
                    e1, e2 = st.columns(2)
                    with e1:
                        if st.button("Зберегти", key=f"cu_save_name_{lid}",
                                     type="primary", use_container_width=True):
                            if rename_lesson(user_id, lid, new_name):
                                st.session_state.pop(f"cu_edit_open_{lid}", None)
                                st.rerun()
                    with e2:
                        if st.button("Скасувати", key=f"cu_cancel_name_{lid}",
                                     use_container_width=True):
                            st.session_state.pop(f"cu_edit_open_{lid}", None)
                            st.rerun()

                if st.session_state.get(f"cu_del_confirm_{lid}"):
                    st.warning(f"Видалити урок «{row['lesson_name']}» назавжди?")
                    d1, d2 = st.columns(2)
                    with d1:
                        if st.button("Так, видалити", type="primary",
                                     key=f"cu_del_yes_{lid}",
                                     use_container_width=True):
                            delete_lesson(user_id, lid)
                            st.session_state.pop(f"cu_del_confirm_{lid}", None)
                            st.rerun()
                    with d2:
                        if st.button("Скасувати", key=f"cu_del_no_{lid}",
                                     use_container_width=True):
                            st.session_state.pop(f"cu_del_confirm_{lid}", None)
                            st.rerun()

    # ── Create new lesson ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("➕ Створити новий урок", expanded=lessons_df.empty):
        lesson_name = st.text_input("Назва уроку",
                                     placeholder="наприклад, Подорож у Париж",
                                     key="cu_new_lesson_name")

        st.caption(
            f"Список фраз — по одній на рядок у форматі **{native} = {target}**. "
            f"Приклад:\n\n"
            f"`Я хочу каву = I want a coffee`\n"
            f"`Скільки коштує? = How much is it?`"
        )
        text = st.text_area(
            f"Пари фраз ({native} = {target})",
            height=200,
            key="cu_new_pairs",
            placeholder=(
                f"{native} = {target}\n"
                f"...\n"
                f"...\n"
            ),
        )

        # Live preview of what will be saved
        pairs = parse_pairs_text(text, sep="=")
        if pairs:
            st.caption(f"Розпізнано {len(pairs)} пар(и):")
            preview_html = ""
            for i, (nat, tgt) in enumerate(pairs[:8], start=1):
                preview_html += (
                    f'<div style="display:flex;gap:14px;padding:6px 10px;'
                    f'background:var(--mova-card);border-bottom:1px solid var(--mova-line)">'
                    f'<span style="color:var(--mova-indigo-ink);min-width:28px;'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:.72rem">{i:02d}</span>'
                    f'<span style="flex:1;color:var(--mova-ink)">{nat}</span>'
                    f'<span style="flex:1;color:#ffffff;font-weight:500">{tgt}</span>'
                    f'</div>'
                )
            if len(pairs) > 8:
                preview_html += (
                    f'<div style="padding:6px 10px;color:var(--mova-ink-3);font-size:.78rem">'
                    f'... та ще {len(pairs)-8} пар</div>'
                )
            st.markdown(
                f'<div style="border-radius:10px;overflow:hidden;'
                f'background:var(--mova-card);border:1px solid var(--mova-line)">{preview_html}</div>',
                unsafe_allow_html=True,
            )
        elif text.strip():
            st.warning("Жодної валідної пари не знайдено. "
                       "Перевірте, що використовуєте `=` між фразами.")

        save_disabled = (not pairs) or (not user_id)
        if st.button("💾 Зберегти урок", type="primary",
                     use_container_width=True,
                     disabled=save_disabled, key="cu_save_new"):
            try:
                lid = add_lesson(user_id, lesson_name, native, target, pairs)
                st.success(f"✓ Урок створено (id {lid}, {len(pairs)} фраз).")
                st.session_state.pop("cu_new_pairs", None)
                st.session_state.pop("cu_new_lesson_name", None)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Не вдалося зберегти: {e}")


# ─── Start a lesson — push state and let grammar.main() run the 8 steps ──

def _start_lesson(user_id: str, lesson_id: int,
                   native: str, target: str, lang_pair: str):
    lesson_df = get_lesson_phrases(user_id, lesson_id)
    if lesson_df.empty:
        st.error("У цьому уроці немає фраз.")
        return

    # Reset any previous lesson state
    for k in list(st.session_state):
        if k.startswith(("s1_", "s2_", "s3_", "s4_", "s5_", "s6_", "s7_", "s8_",
                          "mic_", "up_")):
            del st.session_state[k]
    st.session_state.pop("_progress_saved", None)
    st.session_state.pop("_last_saved_progress", None)

    st.session_state.update({
        "practice_module": "custom",
        "session":         LessonSession(user_id, lesson_df, lesson_id,
                                          native, target,
                                          language_pair=lang_pair),
        "lesson_step":     1,
        "tts_lang":        TTS_LANG.get(target, "en"),
        "wh_lang":         WHISPER_LANG.get(target),
        "lang_pair":       lang_pair,
    })
    st.rerun()


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    """Custom practice entry point.

    If a lesson has been started (``lesson_step`` in session state), defer
    to grammar.main() which already implements the 8-step flow. Otherwise
    show our setup / management screen.
    """
    if "lesson_step" in st.session_state and "session" in st.session_state:
        # Re-use grammar's main() — it reads practice_module and renders
        # the steps + sidebar nav.
        grammar_app.main(module="custom")
        return

    # No active lesson — show our setup screen
    st.session_state["practice_module"] = "custom"
    render_setup()


if __name__ == "__main__":
    main()
