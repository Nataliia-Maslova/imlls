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

import json
import streamlit as st
import streamlit.components.v1 as components

ROOT        = Path(__file__).parent
APP_IMG_DIR = ROOT / "static" / "app_images"
sys.path.insert(0, str(ROOT))

from engine.custom_store import (
    add_lesson, delete_lesson, get_lesson_phrases,
    list_user_lessons, parse_pairs_text, rename_lesson,
)
from engine.loader  import TTS_LANG, WHISPER_LANG
from engine.session import LessonSession

import grammar as grammar_app  # we reuse its 8-step machinery


LANGUAGES = ["English", "Ukrainian", "Spanish", "Korean"]


# ─── i18n strings ─────────────────────────────────────────────────────────

_I18N: dict[str, dict[str, str]] = {
    "Ukrainian": {
        "module_label":     "Модуль",
        "main_menu":        "🏠 Головне меню",
        "title":            "Мої фрази",
        "subtitle":         "Створи свій урок із власних фраз — і тренуй той самий 8-кроковий цикл.",
        "your_name":        "👤 Ваше ім'я",
        "native_label":     "🌐 Рідна",
        "target_label":     "🎯 Цільова",
        "your_lessons":     "📚 Ваші уроки ({native} → {target})",
        "no_lessons":       "Поки що жодного уроку для цієї пари. Створіть перший нижче ↓",
        "manage_lessons":   "✏️ Керування уроками (перейменування / видалення)",
        "phrases_count":    "· {n} фраз",
        "rename_btn":       "✏ Перейменувати",
        "delete_btn":       "🗑 Видалити",
        "new_name_label":   "Нова назва уроку",
        "save_btn":         "Зберегти",
        "cancel_btn":       "Скасувати",
        "delete_confirm":   "Видалити урок «{name}» назавжди?",
        "delete_yes":       "Так, видалити",
        "create_expander":  "➕ Створити новий урок",
        "lesson_name":      "Назва уроку",
        "lesson_name_ph":   "наприклад, Подорож у Париж",
        "pairs_hint":       "Список фраз — по одній на рядок у форматі **{native} = {target}**. Приклад:\n\n`Я хочу каву = I want a coffee`\n`Скільки коштує? = How much is it?`",
        "pairs_label":      "Пари фраз ({native} = {target})",
        "pairs_ph":         "{native} = {target}\n...\n...",
        "recognized":       "Розпізнано {n} пар(и):",
        "more_pairs":       "... та ще {n} пар",
        "no_pairs":         "Жодної валідної пари не знайдено. Перевірте, що використовуєте `=` між фразами.",
        "save_lesson":      "💾 Зберегти урок",
        "saved_ok":         "✓ Урок створено (id {lid}, {n} фраз).",
        "no_phrases_err":   "У цьому уроці немає фраз.",
    },
    "English": {
        "module_label":     "Module",
        "main_menu":        "🏠 Main menu",
        "title":            "My Phrases",
        "subtitle":         "Create your own lesson from custom phrases and practice the same 8-step cycle.",
        "your_name":        "👤 Your name",
        "native_label":     "🌐 Native",
        "target_label":     "🎯 Target",
        "your_lessons":     "📚 Your lessons ({native} → {target})",
        "no_lessons":       "No lessons yet for this pair. Create your first one below ↓",
        "manage_lessons":   "✏️ Manage lessons (rename / delete)",
        "phrases_count":    "· {n} phrases",
        "rename_btn":       "✏ Rename",
        "delete_btn":       "🗑 Delete",
        "new_name_label":   "New lesson name",
        "save_btn":         "Save",
        "cancel_btn":       "Cancel",
        "delete_confirm":   "Delete lesson «{name}» permanently?",
        "delete_yes":       "Yes, delete",
        "create_expander":  "➕ Create new lesson",
        "lesson_name":      "Lesson name",
        "lesson_name_ph":   "e.g. Trip to Paris",
        "pairs_hint":       "One phrase per line in the format **{native} = {target}**. Example:\n\n`I want coffee = Я хочу каву`\n`How much is it? = Скільки коштує?`",
        "pairs_label":      "Phrase pairs ({native} = {target})",
        "pairs_ph":         "{native} = {target}\n...\n...",
        "recognized":       "Recognized {n} pair(s):",
        "more_pairs":       "... and {n} more",
        "no_pairs":         "No valid pairs found. Make sure you use `=` between phrases.",
        "save_lesson":      "💾 Save lesson",
        "saved_ok":         "✓ Lesson created (id {lid}, {n} phrases).",
        "no_phrases_err":   "No phrases in this lesson.",
    },
    "Spanish": {
        "module_label":     "Módulo",
        "main_menu":        "🏠 Menú principal",
        "title":            "Mis frases",
        "subtitle":         "Crea tu propia lección con frases personales y practica el ciclo de 8 pasos.",
        "your_name":        "👤 Tu nombre",
        "native_label":     "🌐 Idioma nativo",
        "target_label":     "🎯 Idioma objetivo",
        "your_lessons":     "📚 Tus lecciones ({native} → {target})",
        "no_lessons":       "Aún no hay lecciones para este par. Crea la primera abajo ↓",
        "manage_lessons":   "✏️ Gestionar lecciones (renombrar / eliminar)",
        "phrases_count":    "· {n} frases",
        "rename_btn":       "✏ Renombrar",
        "delete_btn":       "🗑 Eliminar",
        "new_name_label":   "Nuevo nombre de lección",
        "save_btn":         "Guardar",
        "cancel_btn":       "Cancelar",
        "delete_confirm":   "¿Eliminar la lección «{name}» permanentemente?",
        "delete_yes":       "Sí, eliminar",
        "create_expander":  "➕ Crear nueva lección",
        "lesson_name":      "Nombre de la lección",
        "lesson_name_ph":   "p.ej. Viaje a París",
        "pairs_hint":       "Una frase por línea en el formato **{native} = {target}**. Ejemplo:\n\n`Quiero un café = I want a coffee`\n`¿Cuánto cuesta? = How much is it?`",
        "pairs_label":      "Pares de frases ({native} = {target})",
        "pairs_ph":         "{native} = {target}\n...\n...",
        "recognized":       "Se reconocieron {n} par(es):",
        "more_pairs":       "... y {n} más",
        "no_pairs":         "No se encontraron pares válidos. Asegúrate de usar `=` entre las frases.",
        "save_lesson":      "💾 Guardar lección",
        "saved_ok":         "✓ Lección creada (id {lid}, {n} frases).",
        "no_phrases_err":   "Esta lección no tiene frases.",
    },
    "Korean": {
        "module_label":     "모듈",
        "main_menu":        "🏠 메인 메뉴",
        "title":            "내 문장",
        "subtitle":         "나만의 문장으로 수업을 만들고 8단계 사이클로 연습하세요.",
        "your_name":        "👤 이름",
        "native_label":     "🌐 모국어",
        "target_label":     "🎯 학습 언어",
        "your_lessons":     "📚 내 수업 ({native} → {target})",
        "no_lessons":       "이 언어 쌍에 대한 수업이 없습니다. 아래에서 첫 수업을 만드세요 ↓",
        "manage_lessons":   "✏️ 수업 관리 (이름 변경 / 삭제)",
        "phrases_count":    "· {n}개 문장",
        "rename_btn":       "✏ 이름 변경",
        "delete_btn":       "🗑 삭제",
        "new_name_label":   "새 수업 이름",
        "save_btn":         "저장",
        "cancel_btn":       "취소",
        "delete_confirm":   "수업 «{name}»을(를) 영구적으로 삭제할까요?",
        "delete_yes":       "예, 삭제",
        "create_expander":  "➕ 새 수업 만들기",
        "lesson_name":      "수업 이름",
        "lesson_name_ph":   "예: 파리 여행",
        "pairs_hint":       "**{native} = {target}** 형식으로 한 줄에 하나씩 입력하세요. 예:\n\n`커피 주세요 = I want a coffee`\n`얼마예요? = How much is it?`",
        "pairs_label":      "문장 쌍 ({native} = {target})",
        "pairs_ph":         "{native} = {target}\n...\n...",
        "recognized":       "{n}개 쌍을 인식했습니다:",
        "more_pairs":       "... 외 {n}개",
        "no_pairs":         "유효한 쌍을 찾을 수 없습니다. 문장 사이에 `=`를 사용했는지 확인하세요.",
        "save_lesson":      "💾 수업 저장",
        "saved_ok":         "✓ 수업이 생성되었습니다 (id {lid}, {n}개 문장).",
        "no_phrases_err":   "이 수업에 문장이 없습니다.",
    },
}

def _t(key: str, **kwargs) -> str:
    """Return localized string for the current native language."""
    lang = st.session_state.get("launcher_native", "Ukrainian")
    strings = _I18N.get(lang, _I18N["Ukrainian"])
    tmpl = strings.get(key, _I18N["Ukrainian"].get(key, key))
    return tmpl.format(**kwargs) if kwargs else tmpl


# ─── Styling reused from grammar module ───────────────────────────────────

def _inject_css():
    grammar_app._inject_css()


# ─── Setup screen ─────────────────────────────────────────────────────────

def render_setup():
    _inject_css()

    # ── Query-params bridge: wave clicked a lesson ────────────────────────────
    _qp = st.query_params
    if "vnav_lesson" in _qp:
        try:
            from urllib.parse import unquote as _uq
            _qp_lid    = int(_qp["vnav_lesson"])
            _qp_native = _uq(_qp.get("vnav_native", "Ukrainian"))
            _qp_target = _uq(_qp.get("vnav_target", "English"))
            _qp_user   = _uq(_qp.get("vnav_user",   "student1"))
            if _qp_native not in LANGUAGES:
                _qp_native = "Ukrainian"
            if _qp_target not in LANGUAGES or _qp_target == _qp_native:
                _qp_target = next(l for l in LANGUAGES if l != _qp_native)
            st.query_params.clear()
            _qp_lp = f"{WHISPER_LANG.get(_qp_native,'?')}-{WHISPER_LANG.get(_qp_target,'?')}-custom"
            _start_lesson(_qp_user, _qp_lid, _qp_native, _qp_target, _qp_lp)
        except Exception:
            st.query_params.clear()


    st.markdown(f"""
    <div style="text-align:center;padding:32px 0 16px">
      <h1 style="color:var(--mova-ink);font-weight:700;margin:0;font-size:2rem">{_t("title")}</h1>
    </div>
    """, unsafe_allow_html=True)

    _my_phrases_banner = APP_IMG_DIR / "my_phrases_banner.jpg"
    if _my_phrases_banner.exists():
        _, _mid, _ = st.columns([1, 2, 1])
        with _mid:
            st.image(str(_my_phrases_banner), use_container_width=True)

    # ── Identity + language pair ──────────────────────────────────────────
    default_user   = st.session_state.get("launcher_user",   "student1")
    default_native = st.session_state.get("launcher_native", "Ukrainian")
    default_target = st.session_state.get("launcher_target", "English")

    cA, cB, cC = st.columns([2, 1.3, 1.3])
    with cA:
        user_id = st.text_input(_t("your_name"), value=default_user, key="cu_user")
    with cB:
        if default_native not in LANGUAGES:
            default_native = "Ukrainian"
        native = st.selectbox(_t("native_label"), LANGUAGES,
                              index=LANGUAGES.index(default_native), key="cu_native")
    with cC:
        target_opts = [l for l in LANGUAGES if l != native]
        tdef = (target_opts.index(default_target)
                if default_target in target_opts else 0)
        target = st.selectbox(_t("target_label"), target_opts, index=tdef, key="cu_target")

    lang_pair = f"{WHISPER_LANG.get(native,'?')}-{WHISPER_LANG.get(target,'?')}-custom"

    # ── Existing lessons for this user + pair ─────────────────────────────
    lessons_df = list_user_lessons(user_id, native_lang=native, target_lang=target)

    st.markdown("---")
    st.markdown(f"### {_t('your_lessons', native=native, target=target)}")
    if lessons_df.empty:
        st.info(_t("no_lessons"))
    else:
        # ── Wave navigator ────────────────────────────────────────────────────
        _cu_lp = f"{WHISPER_LANG.get(native,'?')}-{WHISPER_LANG.get(target,'?')}-custom"
        _cu_lessons    = [int(row["lesson_id"])   for _, row in lessons_df.iterrows()]
        _cu_names_dict = {int(row["lesson_id"]): str(row["lesson_name"])
                          for _, row in lessons_df.iterrows()}
        _cu_counts_dict= {int(row["lesson_id"]): int(row["phrases"])
                          for _, row in lessons_df.iterrows()}

        from engine.picker import _render_wave_plotly

        _cu_clicked = _render_wave_plotly(
            lessons=_cu_lessons,
            lesson_names=_cu_names_dict,
            lesson_counts=_cu_counts_dict,
            default_lid=_cu_lessons[0] if _cu_lessons else None,
            resume_step=1,
            key_suffix=f"custom_{native}_{target}",
            show_lesson_image=False,
        )
        if _cu_clicked is not None:
            _start_lesson(user_id, _cu_clicked, native, target, _cu_lp)
        else:
            # Fallback: circular native Streamlit buttons
            st.markdown("""
<style>
[data-testid="stHorizontalBlock"] [data-testid="stButton"] button {
    border-radius: 50% !important;
    width: 76px !important; height: 76px !important;
    min-width: 76px !important; padding: 3px !important;
    font-size: 0.57rem !important; line-height: 1.22 !important;
    white-space: pre-wrap !important; overflow: hidden !important;
    text-align: center !important;
}
</style>""", unsafe_allow_html=True)
            _cu_emojis = ['📖','🌟','💡','🎯','🔤','🗣️','✏️','📝','🎓','💬','🔑','🏆']
            _cu_n = len(_cu_lessons)
            _cu_rows_per_row = 10
            for _cu_rs in range(0, _cu_n, _cu_rows_per_row):
                _cu_chunk_ids = _cu_lessons[_cu_rs:_cu_rs + _cu_rows_per_row]
                _cu_cols = st.columns(len(_cu_chunk_ids))
                for _cu_ci, (_cu_col, _cu_lid) in enumerate(zip(_cu_cols, _cu_chunk_ids)):
                    _cu_name  = _cu_names_dict.get(_cu_lid, f"Lesson {_cu_lid}")
                    _cu_cnt   = _cu_counts_dict.get(_cu_lid, 0)
                    _cu_gidx  = _cu_rs + _cu_ci
                    _cu_emoji = _cu_emojis[_cu_gidx % len(_cu_emojis)]
                    _cu_short = (_cu_name[:10] + "…") if len(_cu_name) > 10 else _cu_name
                    with _cu_col:
                        if st.button(
                            f"{_cu_emoji}\n{_cu_short}",
                            key=f"cu_wv_{_cu_lid}_{_cu_gidx}",
                            type="secondary",
                            use_container_width=False,
                            help=f"{_cu_name} · {_cu_cnt} phrases",
                        ):
                            _start_lesson(user_id, _cu_lid, native, target, _cu_lp)

        # ── Manage lessons (edit / delete) ──────────────────────────────────────
        with st.expander(_t("manage_lessons"), expanded=False):
            for _, row in lessons_df.iterrows():
                lid = int(row["lesson_id"])
                with st.container():
                    c1, c2, c3 = st.columns([5, 1.5, 1.5])
                    with c1:
                        st.markdown(
                            f"**{row['lesson_name']}** "
                            f"<span style='color:var(--mova-ink-3);font-size:.8rem'>"
                            f"{_t('phrases_count', n=row['phrases'])}</span>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        if st.button(_t("rename_btn"), key=f"cu_edit_{lid}",
                                     use_container_width=True):
                            st.session_state[f"cu_edit_open_{lid}"] = True
                    with c3:
                        if st.button(_t("delete_btn"), key=f"cu_del_{lid}",
                                     use_container_width=True):
                            st.session_state[f"cu_del_confirm_{lid}"] = True

                if st.session_state.get(f"cu_edit_open_{lid}"):
                    new_name = st.text_input(
                        _t("new_name_label"),
                        value=row["lesson_name"],
                        key=f"cu_new_name_{lid}",
                    )
                    e1, e2 = st.columns(2)
                    with e1:
                        if st.button(_t("save_btn"), key=f"cu_save_name_{lid}",
                                     type="primary", use_container_width=True):
                            if rename_lesson(user_id, lid, new_name):
                                st.session_state.pop(f"cu_edit_open_{lid}", None)
                                st.rerun()
                    with e2:
                        if st.button(_t("cancel_btn"), key=f"cu_cancel_name_{lid}",
                                     use_container_width=True):
                            st.session_state.pop(f"cu_edit_open_{lid}", None)
                            st.rerun()

                if st.session_state.get(f"cu_del_confirm_{lid}"):
                    st.warning(_t("delete_confirm", name=row["lesson_name"]))
                    d1, d2 = st.columns(2)
                    with d1:
                        if st.button(_t("delete_yes"), type="primary",
                                     key=f"cu_del_yes_{lid}",
                                     use_container_width=True):
                            delete_lesson(user_id, lid)
                            st.session_state.pop(f"cu_del_confirm_{lid}", None)
                            st.rerun()
                    with d2:
                        if st.button(_t("cancel_btn"), key=f"cu_del_no_{lid}",
                                     use_container_width=True):
                            st.session_state.pop(f"cu_del_confirm_{lid}", None)
                            st.rerun()

    # ── Create new lesson ────────────────────────────────────────
    st.markdown("---")
    with st.expander(_t("create_expander"), expanded=lessons_df.empty):
        lesson_name = st.text_input(_t("lesson_name"),
                                     placeholder=_t("lesson_name_ph"),
                                     key="cu_new_lesson_name")

        st.caption(_t("pairs_hint", native=native, target=target))
        text = st.text_area(
            _t("pairs_label", native=native, target=target),
            height=200,
            key="cu_new_pairs",
            placeholder=_t("pairs_ph", native=native, target=target),
        )

        pairs = parse_pairs_text(text, sep="=")
        if pairs:
            st.caption(_t("recognized", n=len(pairs)))
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
                    f'{_t("more_pairs", n=len(pairs)-8)}</div>'
                )
            st.markdown(
                f'<div style="border-radius:10px;overflow:hidden;'
                f'background:var(--mova-card);border:1px solid var(--mova-line)">{preview_html}</div>',
                unsafe_allow_html=True,
            )
        elif text.strip():
            st.warning(_t("no_pairs"))

        save_disabled = (not pairs) or (not user_id)
        if st.button(_t("save_lesson"), type="primary",
                     use_container_width=True,
                     disabled=save_disabled, key="cu_save_new"):
            try:
                lid = add_lesson(user_id, lesson_name, native, target, pairs)
                st.success(_t("saved_ok", lid=lid, n=len(pairs)))
                st.session_state.pop("cu_new_pairs", None)
                st.session_state.pop("cu_new_lesson_name", None)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"{e}")


# ─── Start a lesson — push state and let grammar.main() run the 8 steps ──

def _start_lesson(user_id: str, lesson_id: int,
                   native: str, target: str, lang_pair: str):
    lesson_df = get_lesson_phrases(user_id, lesson_id)
    if lesson_df.empty:
        st.error(_t("no_phrases_err"))
        return

    for k in list(st.session_state):
        if k.startswith(("s1_", "s2_", "s3_", "s4_", "s5_", "s6_", "s7_", "s8_",
                          "mic_", "up_")):
            del st.session_state[k]
    st.session_state.pop("_progress_saved", None)
    st.session_state.pop("_last_saved_progress", None)

    from engine.loader import TTS_LANG, WHISPER_LANG
    from engine.session import LessonSession
    st.session_state.update({
        "practice_module": "custom",
        "session":         LessonSession(user_id, lesson_df, lesson_id,
                                          native, target,
                                          language_pair=lang_pair),
        "lesson_step":     1,
        "tts_lang":        TTS_LANG.get(target, "en"),
        "wh_lang":         WHISPER_LANG.get(target, "en"),
        "active_module":   "custom",
    })
    st.query_params["module"] = "custom"
    st.rerun()


def main():
    grammar_app.main(module="custom")


if __name__ == "__main__":
    main()
