"""
engine/character_widget.py — Streamlit-компонент для відображення персонажів.

Мова підхоплюється автоматично з sess.state.native_lang (пріоритет)
або launcher_native. Якщо нічого немає — "English".

Використання
------------
    from engine.character_widget import show_character

    show_character("mark", "on_mistake")
    show_character("natalia", "on_lesson_complete")
    show_character("ai_bot", "feedback", score=87, corrections=3)
    show_character("polyglot", "rare_bonus")
"""
from __future__ import annotations
import base64
from pathlib import Path

import streamlit as st
from engine.characters import get_phrase

# ── Кольорова схема ──────────────────────────────────────────────────────────
CHARACTER_COLORS: dict[str, dict[str, str]] = {
    "natalia":  {"bg": "#FFF3E0", "border": "#FF8C00", "name": "#E65100"},
    "mark":     {"bg": "#E3F2FD", "border": "#1976D2", "name": "#0D47A1"},
    "sophie":   {"bg": "#FCE4EC", "border": "#E91E63", "name": "#880E4F"},
    "ai_bot":   {"bg": "#E8F5E9", "border": "#43A047", "name": "#1B5E20"},
    "polyglot": {"bg": "#EDE7F6", "border": "#7B1FA2", "name": "#4A148C"},
}
_DEFAULT_COLORS: dict[str, str] = {
    "bg": "#F5F5F5", "border": "#9E9E9E", "name": "#212121",
}

_EMOJI_FALLBACK: dict[str, str] = {
    "natalia":  "👩‍🏫",
    "mark":     "😰",
    "sophie":   "📱",
    "ai_bot":   "🤖",
    "polyglot": "🎭",
}

# Папка з PNG-файлами персонажів
_ASSETS_ROOT = Path(__file__).parent.parent / "assets" / "characters"


@st.cache_data(show_spinner=False)
def _img_to_b64(img_path: str) -> str:
    """Повертає base64-рядок PNG для вбудовування в HTML <img>.
    Кешується, щоб не читати файл при кожному рендері.
    Використання повної роздільності через HTML дозволяє браузеру
    коректно масштабувати зображення на high-DPI (Retina / мобільні) екранах,
    уникаючи розмитості, яка виникає при серверному стисненні через st.image.
    """
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _get_lang() -> str:
    """
    Пріоритет: sess.state.native_lang > launcher_native > "English"
    Повертає повну назву мови: "Spanish", "Ukrainian" тощо.
    """
    sess = st.session_state.get("session")
    if sess is not None:
        lang = getattr(getattr(sess, "state", None), "native_lang", None)
        if lang:
            return lang
    return st.session_state.get("launcher_native", "English")


def show_character(
    character_key: str,
    category: str,
    lang: str | None = None,
    **kwargs,
) -> None:
    """
    Відображає картку персонажа з фразою.

    Параметри
    ---------
    character_key : "natalia" | "mark" | "sophie" | "ai_bot" | "polyglot"
    category      : "on_mistake" | "on_lesson_complete" | "motivation" | ...
    lang          : явно задати мову (якщо None — береться з сесії)
    **kwargs      : підстановки {score}, {corrections} тощо
    """
    if lang is None:
        lang = _get_lang()

    data = get_phrase(character_key, category, lang=lang, **kwargs)
    if data is None:
        return

    colors = CHARACTER_COLORS.get(character_key, _DEFAULT_COLORS)
    img_path = _ASSETS_ROOT / Path(data["image"]).name if data.get("image") else None
    img_exists = img_path is not None and img_path.exists()

    col_img, col_text = st.columns([1, 4])

    with col_img:
        if img_exists:
            # Відображаємо через HTML <img> з base64, щоб браузер сам
            # масштабував повну роздільність — це усуває розмитість
            # на мобільних Retina-екранах (яка виникає при st.image з width=90,
            # бо Streamlit стискає зображення на сервері до 90px).
            b64 = _img_to_b64(str(img_path))
            st.markdown(
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:90px;height:auto;display:block;margin:0 auto;" '
                f'loading="lazy">',
                unsafe_allow_html=True,
            )
        else:
            emoji = _EMOJI_FALLBACK.get(character_key, "🙂")
            st.markdown(
                "<div style='font-size:3rem;text-align:center;'>"
                + emoji + "</div>",
                unsafe_allow_html=True,
            )
        name_html = (
            "<p style='text-align:center;font-weight:600;"
            "color:" + colors["name"] + ";margin:2px 0 0 0;font-size:.78rem;'>"
            + data["name"] + "</p>"
        )
        st.markdown(name_html, unsafe_allow_html=True)

    with col_text:
        phrase = data["phrase"].replace("<", "&lt;").replace(">", "&gt;")
        bubble_html = (
            "<div style='"
            "background:" + colors["bg"] + ";"
            "border-left:4px solid " + colors["border"] + ";"
            "border-radius:0 12px 12px 0;"
            "padding:14px 18px;"
            "margin-top:8px;"
            "font-size:.95rem;"
            "color:#333;"
            "line-height:1.5;"
            "box-shadow:0 2px 6px rgba(0,0,0,.07);"
            "'>&#128172; " + phrase + "</div>"
        )
        st.markdown(bubble_html, unsafe_allow_html=True)
