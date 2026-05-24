"""
engine/characters.py — завантаження фраз персонажів із YAML.

Використання:
    from engine.characters import get_phrase

    data = get_phrase("natalia", "on_mistake", lang="uk")
    # -> {"name": "Наталія", "image": "assets/characters/natalia.png", "phrase": "..."}

Логіка вибору фрази
-------------------
Функція збирає фрази з УСІХ категорій персонажа і вибирає рандомно,
відфільтровуючи фрази з незаповненими шаблонними змінними ({score} тощо).
Параметр `category` використовується як пріоритет: якщо є фрази в зазначеній
категорії — вибір відбувається серед них; якщо категорія порожня — з усіх.
"""
from __future__ import annotations
import re
import random
from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).parent.parent / "assets" / "characters" / "phrases.yaml"
_cache = None

LANG_NAME_TO_CODE = {
    "English":   "en",
    "Ukrainian": "uk",
    "Spanish":   "es",
    "Korean":    "ko",
}


def load_characters():
    global _cache
    if _cache is None:
        with open(_YAML_PATH, encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def _resolve_lang(lang):
    if lang in LANG_NAME_TO_CODE:
        return LANG_NAME_TO_CODE[lang]
    if lang in LANG_NAME_TO_CODE.values():
        return lang
    return "en"


def _phrases_for_category(char, category, code):
    by_lang = char.get("phrases", {}).get(category, {})
    return (
        by_lang.get(code)
        or by_lang.get("en")
        or by_lang.get("uk")
        or []
    )


def _all_phrases(char, code):
    result = []
    for cat_data in char.get("phrases", {}).values():
        phrases = (
            cat_data.get(code)
            or cat_data.get("en")
            or cat_data.get("uk")
            or []
        )
        result.extend(phrases)
    return result


def _can_format(phrase, kwargs):
    needed = set(re.findall(r'\{(\w+)\}', phrase))
    return needed.issubset(kwargs.keys())


def get_phrase(character_key, category, lang="uk", **kwargs):
    """
    Повертає словник з ім'ям, зображенням і фразою для вказаного персонажа.

    Параметри
    ---------
    character_key : str  — "natalia", "mark", "sophie", "ai_bot", "polyglot"
    category      : str  — пріоритетна категорія; якщо порожня — рандом з усіх
    lang          : str  — "uk"/"en"/"es"/"ko" або повна назва
    **kwargs             — підстановки {score}, {corrections} тощо
    """
    chars = load_characters()
    char = chars.get(character_key)
    if not char:
        return None

    code = _resolve_lang(lang)

    phrases = _phrases_for_category(char, category, code)

    if not phrases:
        phrases = _all_phrases(char, code)

    if not phrases:
        return None

    valid = [p for p in phrases if _can_format(p, kwargs)]
    if not valid:
        valid = phrases

    phrase = random.choice(valid)

    if kwargs:
        try:
            phrase = phrase.format(**kwargs)
        except (KeyError, ValueError):
            pass

    name_map = char.get("name", {})
    name = name_map.get(code) or name_map.get("en") or character_key

    return {
        "name":   name,
        "image":  char.get("image", ""),
        "phrase": phrase,
    }
