"""
engine/gamification.py — Streak · XP · Levels · Badges · Time

Public API (all functions are safe to call even if the CSV is missing):
  load_stats(user_id)         → dict
  on_step_complete(user_id, step, similarity) → result dict
  on_lesson_complete(user_id) → result dict
  get_level(xp)               → (num, name, pct_in_level, xp_to_next)
  tick_time(user_id)          → adds elapsed time to total_time_minutes
  sidebar_widget(user_id)     → renders compact streak/xp/time block via st.markdown
"""
from __future__ import annotations

import csv
import json
import time as _time
from datetime import date, datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
GAMI_CSV = DATA_DIR / "gamification.csv"

_COLUMNS = [
    "user_id", "streak_current", "streak_max", "streak_last_date",
    "xp_total", "lessons_completed", "daily_xp_date", "daily_xp", "badges",
    "total_time_minutes",
]

# Maximum minutes counted per single time-flush interval.
# Prevents counting idle time if the user left the tab open.
_MAX_IDLE_MINUTES = 5

# ─── XP table ────────────────────────────────────────────────────────────────
#  step_xp: base XP for completing each step (1-8)
STEP_XP = {1: 5, 2: 10, 3: 10, 4: 15, 5: 15, 6: 10, 7: 20, 8: 15}
LESSON_BONUS_XP   = 30   # awarded when whole lesson is done
HIGH_ACCURACY_XP  = 5    # extra if similarity > 0.80

# ─── Level thresholds ────────────────────────────────────────────────────────
# (min_xp, level_num) — names are localised in _LEVEL_NAMES below
_LEVELS = [
    (0,    1),
    (100,  2),
    (250,  3),
    (500,  4),
    (900,  5),
    (1500, 6),
    (2500, 7),
]

# level names by native-language code (fallback → en)
_LEVEL_NAMES: dict[str, list[str]] = {
    "en": ["Beginner", "Learner", "Student", "Practitioner", "Specialist", "Master", "Expert"],
    "uk": ["Початківець", "Учень", "Студент", "Практик", "Спеціаліст", "Майстер", "Експерт"],
    "es": ["Principiante", "Aprendiz", "Estudiante", "Practicante", "Especialista", "Maestro", "Experto"],
    "ko": ["초보자", "학습자", "학생", "실습자", "전문가", "달인", "전문가"],
}

# map Streamlit launcher_native → lang code
_NATIVE_TO_CODE = {
    "English":   "en",
    "Ukrainian": "uk",
    "Spanish":   "es",
    "Korean":    "ko",
}

def _level_name(level_num: int, lang: str = "en") -> str:
    names = _LEVEL_NAMES.get(lang, _LEVEL_NAMES["en"])
    return names[min(level_num - 1, len(names) - 1)]

# ─── UI strings ──────────────────────────────────────────────────────────────
_UI: dict[str, dict[str, str]] = {
    "en": {
        "level":       "Level",
        "day":         "day",
        "days":        "days",
        "no_streak":   "0 days",
        "xp_next":     "+{n} XP to next level",
        "xp_max":      "Max level!",
        "time_label":  "Study time",
        "time_h":      "h",
        "time_m":      "m",
        "time_zero":   "< 1 m",
    },
    "uk": {
        "level":       "Рівень",
        "day":         "день",
        "days":        "дн",
        "no_streak":   "0 днів",
        "xp_next":     "+{n} XP до наст. рівня",
        "xp_max":      "Максимальний рівень!",
        "time_label":  "Час навчання",
        "time_h":      "год",
        "time_m":      "хв",
        "time_zero":   "< 1 хв",
    },
    "es": {
        "level":       "Nivel",
        "day":         "día",
        "days":        "días",
        "no_streak":   "0 días",
        "xp_next":     "+{n} XP al siguiente nivel",
        "xp_max":      "¡Nivel máximo!",
        "time_label":  "Tiempo de estudio",
        "time_h":      "h",
        "time_m":      "min",
        "time_zero":   "< 1 min",
    },
    "ko": {
        "level":       "레벨",
        "day":         "일",
        "days":        "일",
        "no_streak":   "0일",
        "xp_next":     "다음 레벨까지 +{n} XP",
        "xp_max":      "최고 레벨!",
        "time_label":  "학습 시간",
        "time_h":      "시간",
        "time_m":      "분",
        "time_zero":   "< 1 분",
    },
}

def _t(key: str, lang: str = "en", **kw) -> str:
    """Fetch a UI string for the given language, format with kwargs."""
    s = _UI.get(lang, _UI["en"]).get(key, _UI["en"].get(key, key))
    return s.format(**kw) if kw else s

# ─── Badge definitions ───────────────────────────────────────────────────────
# Badges are keyed by id; display strings are localised at render time.
# (id, emoji, name_key, desc_key)  — resolved via _BADGE_STRINGS below
BADGE_DEFS_RAW: list[tuple[str, str]] = [
    ("first_lesson", "🌱"),
    ("streak_3",     "🔥"),
    ("streak_7",     "🔥"),
    ("streak_30",    "🔥"),
    ("level_2",      "⭐"),
    ("level_4",      "🌟"),
    ("level_6",      "💫"),
    ("lessons_5",    "📚"),
    ("lessons_10",   "🎓"),
    ("lessons_25",   "🏆"),
    ("daily_100",    "⚡"),
]

_BADGE_STRINGS: dict[str, dict[str, tuple[str, str]]] = {
    # lang → badge_id → (name, desc)
    "en": {
        "first_lesson": ("First step",      "Completed your first lesson"),
        "streak_3":     ("3 days",           "3-day streak"),
        "streak_7":     ("One week",         "7-day streak"),
        "streak_30":    ("One month",        "30-day streak"),
        "level_2":      ("Level 2",          "Reached Learner level"),
        "level_4":      ("Level 4",          "Reached Practitioner level"),
        "level_6":      ("Level 6",          "Reached Master level"),
        "lessons_5":    ("5 lessons",        "Completed 5 lessons"),
        "lessons_10":   ("10 lessons",       "Completed 10 lessons"),
        "lessons_25":   ("25 lessons",       "Completed 25 lessons"),
        "daily_100":    ("Productive",       "100+ XP in one day"),
    },
    "uk": {
        "first_lesson": ("Перший крок",      "Завершив перший урок"),
        "streak_3":     ("3 дні підряд",     "Серія 3 дні поспіль"),
        "streak_7":     ("Тиждень",          "Серія 7 днів поспіль"),
        "streak_30":    ("Місяць",           "Серія 30 днів поспіль"),
        "level_2":      ("Рівень 2",         "Досяг рівня «Учень»"),
        "level_4":      ("Рівень 4",         "Досяг рівня «Практик»"),
        "level_6":      ("Рівень 6",         "Досяг рівня «Майстер»"),
        "lessons_5":    ("5 уроків",         "Завершив 5 уроків"),
        "lessons_10":   ("10 уроків",        "Завершив 10 уроків"),
        "lessons_25":   ("25 уроків",        "Завершив 25 уроків"),
        "daily_100":    ("Продуктивний",     "100+ XP за один день"),
    },
    "es": {
        "first_lesson": ("Primer paso",      "Completaste tu primera lección"),
        "streak_3":     ("3 días",           "Racha de 3 días"),
        "streak_7":     ("Una semana",       "Racha de 7 días"),
        "streak_30":    ("Un mes",           "Racha de 30 días"),
        "level_2":      ("Nivel 2",          "Alcanzaste el nivel Aprendiz"),
        "level_4":      ("Nivel 4",          "Alcanzaste el nivel Practicante"),
        "level_6":      ("Nivel 6",          "Alcanzaste el nivel Maestro"),
        "lessons_5":    ("5 lecciones",      "Completaste 5 lecciones"),
        "lessons_10":   ("10 lecciones",     "Completaste 10 lecciones"),
        "lessons_25":   ("25 lecciones",     "Completaste 25 lecciones"),
        "daily_100":    ("Productivo",       "100+ XP en un día"),
    },
    "ko": {
        "first_lesson": ("첫 걸음",           "첫 번째 수업 완료"),
        "streak_3":     ("3일 연속",          "3일 연속 학습"),
        "streak_7":     ("1주일",             "7일 연속 학습"),
        "streak_30":    ("1개월",             "30일 연속 학습"),
        "level_2":      ("레벨 2",            "학습자 레벨 달성"),
        "level_4":      ("레벨 4",            "실습자 레벨 달성"),
        "level_6":      ("레벨 6",            "달인 레벨 달성"),
        "lessons_5":    ("수업 5개",          "수업 5개 완료"),
        "lessons_10":   ("수업 10개",         "수업 10개 완료"),
        "lessons_25":   ("수업 25개",         "수업 25개 완료"),
        "daily_100":    ("생산적인 날",        "하루 100+ XP"),
    },
}

def get_badge_defs(lang: str = "en") -> list[tuple[str, str, str, str]]:
    """Return (id, emoji, name, desc) tuples localised to lang."""
    strings = _BADGE_STRINGS.get(lang, _BADGE_STRINGS["en"])
    result = []
    for bid, emoji in BADGE_DEFS_RAW:
        name, desc = strings.get(bid, (bid, ""))
        result.append((bid, emoji, name, desc))
    return result

# Keep BADGE_DEFS for backward compat (English)
BADGE_DEFS = get_badge_defs("en")
_BADGE_IDS = {b[0] for b in BADGE_DEFS_RAW}


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _default_stats(user_id: str = "") -> dict:
    return {
        "user_id":             user_id,
        "streak_current":      0,
        "streak_max":          0,
        "streak_last_date":    "",
        "xp_total":            0,
        "lessons_completed":   0,
        "daily_xp_date":       "",
        "daily_xp":            0,
        "badges":              set(),
        "total_time_minutes":  0,
    }


def _read_csv() -> list[dict]:
    if not GAMI_CSV.exists():
        return []
    try:
        with open(GAMI_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _write_csv(rows: list[dict]) -> None:
    with open(GAMI_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            badges_val = r.get("badges", set())
            if isinstance(badges_val, set):
                badges_val = ",".join(sorted(badges_val))
            w.writerow({
                "user_id":            r.get("user_id", ""),
                "streak_current":     int(r.get("streak_current", 0)),
                "streak_max":         int(r.get("streak_max", 0)),
                "streak_last_date":   r.get("streak_last_date", ""),
                "xp_total":           int(r.get("xp_total", 0)),
                "lessons_completed":  int(r.get("lessons_completed", 0)),
                "daily_xp_date":      r.get("daily_xp_date", ""),
                "daily_xp":           int(r.get("daily_xp", 0)),
                "badges":             badges_val,
                "total_time_minutes": round(float(r.get("total_time_minutes", 0)), 1),
            })


def _row_to_stats(row: dict) -> dict:
    badges_raw = row.get("badges", "")
    if isinstance(badges_raw, str):
        badges = {b.strip() for b in badges_raw.split(",") if b.strip()}
    elif isinstance(badges_raw, set):
        badges = badges_raw
    else:
        badges = set()
    return {
        "user_id":             str(row.get("user_id", "")),
        "streak_current":      int(row.get("streak_current", 0) or 0),
        "streak_max":          int(row.get("streak_max", 0) or 0),
        "streak_last_date":    str(row.get("streak_last_date", "") or ""),
        "xp_total":            int(row.get("xp_total", 0) or 0),
        "lessons_completed":   int(row.get("lessons_completed", 0) or 0),
        "daily_xp_date":       str(row.get("daily_xp_date", "") or ""),
        "daily_xp":            int(row.get("daily_xp", 0) or 0),
        "badges":              badges,
        "total_time_minutes":  float(row.get("total_time_minutes", 0) or 0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public: load / save
# ═══════════════════════════════════════════════════════════════════════════

def load_stats(user_id: str) -> dict:
    """Load gamification stats for user_id (or create defaults)."""
    for row in _read_csv():
        if str(row.get("user_id", "")) == user_id:
            return _row_to_stats(row)
    return _default_stats(user_id)


def save_stats(user_id: str, stats: dict) -> None:
    """Upsert stats row for user_id."""
    rows = _read_csv()
    stats["user_id"] = user_id
    for i, row in enumerate(rows):
        if str(row.get("user_id", "")) == user_id:
            rows[i] = stats
            _write_csv(rows)
            return
    rows.append(stats)
    _write_csv(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Public: levels
# ═══════════════════════════════════════════════════════════════════════════

def get_level(xp: int, lang: str = "en") -> tuple[int, str, float, int]:
    """
    Returns (level_num, level_name, pct_progress_in_level, xp_needed_for_next).
    pct_progress_in_level is 0.0-1.0. If max level, xp_needed_for_next=0.
    lang is the native-language code used to localise the level name.
    """
    level_num = _LEVELS[-1][1]
    for threshold, num in _LEVELS:
        if xp < threshold:
            break
        level_num = num

    level_name = _level_name(level_num, lang)

    # find position in current level
    cur_idx = next(i for i, (t, n) in enumerate(_LEVELS) if n == level_num)
    cur_threshold = _LEVELS[cur_idx][0]
    if cur_idx + 1 < len(_LEVELS):
        next_threshold = _LEVELS[cur_idx + 1][0]
        span = next_threshold - cur_threshold
        pct  = min((xp - cur_threshold) / span, 1.0)
        xp_to_next = max(next_threshold - xp, 0)
    else:
        pct        = 1.0
        xp_to_next = 0

    return level_num, level_name, pct, xp_to_next


# ═══════════════════════════════════════════════════════════════════════════
# Internal: streak update
# ═══════════════════════════════════════════════════════════════════════════

def _update_streak(stats: dict) -> dict:
    """Update streak fields based on today's date. Mutates and returns stats."""
    today     = date.today().isoformat()
    last_date = stats.get("streak_last_date", "")

    if last_date == today:
        return stats  # already practised today, no change

    if last_date:
        try:
            delta = (date.today() - date.fromisoformat(last_date)).days
        except ValueError:
            delta = 999
        if delta == 1:
            stats["streak_current"] = int(stats.get("streak_current", 0)) + 1
        elif delta > 1:
            stats["streak_current"] = 1   # streak broken
        # delta == 0 already handled above
    else:
        stats["streak_current"] = 1

    stats["streak_last_date"] = today
    stats["streak_max"] = max(int(stats.get("streak_max", 0)),
                               int(stats["streak_current"]))
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Internal: badge checking
# ═══════════════════════════════════════════════════════════════════════════

def _check_badges(stats: dict, lang: str = "en") -> list[tuple[str, str, str, str]]:
    """Return list of newly earned (id, emoji, name, desc) badge tuples (localised)."""
    earned  = stats.get("badges", set())
    new     = []
    streak  = int(stats.get("streak_current", 0))
    xp      = int(stats.get("xp_total", 0))
    lessons = int(stats.get("lessons_completed", 0))
    daily   = int(stats.get("daily_xp", 0))
    level_num, *_ = get_level(xp)

    conditions: dict[str, bool] = {
        "first_lesson": lessons >= 1,
        "streak_3":     streak  >= 3,
        "streak_7":     streak  >= 7,
        "streak_30":    streak  >= 30,
        "level_2":      level_num >= 2,
        "level_4":      level_num >= 4,
        "level_6":      level_num >= 6,
        "lessons_5":    lessons >= 5,
        "lessons_10":   lessons >= 10,
        "lessons_25":   lessons >= 25,
        "daily_100":    daily   >= 100,
    }

    badge_map = {b[0]: b for b in get_badge_defs(lang)}
    for bid, met in conditions.items():
        if met and bid not in earned:
            new.append(badge_map[bid])
            earned.add(bid)

    stats["badges"] = earned
    return new


# ═══════════════════════════════════════════════════════════════════════════
# Public: time tracking
# ═══════════════════════════════════════════════════════════════════════════

def _format_time(total_minutes: float, lang: str = "en") -> str:
    """Format total minutes into a human-readable string (e.g. '2 год 15 хв')."""
    total_minutes = max(0, int(total_minutes))
    if total_minutes == 0:
        return _t("time_zero", lang)
    hours   = total_minutes // 60
    minutes = total_minutes % 60
    h_lbl   = _t("time_h", lang)
    m_lbl   = _t("time_m", lang)
    if hours > 0 and minutes > 0:
        return f"{hours} {h_lbl} {minutes} {m_lbl}"
    if hours > 0:
        return f"{hours} {h_lbl}"
    return f"{minutes} {m_lbl}"


def tick_time(user_id: str) -> float:
    """
    Measure elapsed time since the last tick and add it to the user's
    total_time_minutes in the CSV.

    Call this at activity points (step complete, lesson complete).
    Uses st.session_state to track the wall-clock start of the current interval.
    Idle intervals longer than _MAX_IDLE_MINUTES are capped to avoid counting
    time when the user left the browser tab open without doing anything.

    Returns the number of minutes added in this call.
    """
    try:
        import streamlit as st
        now = _time.monotonic()
        last = st.session_state.get("_gami_tick_last")

        if last is None:
            # First tick in this Streamlit session — start the clock, add nothing.
            st.session_state["_gami_tick_last"] = now
            return 0.0

        elapsed_minutes = (now - last) / 60.0
        # Cap to avoid counting long idle gaps
        elapsed_minutes = min(elapsed_minutes, _MAX_IDLE_MINUTES)
        st.session_state["_gami_tick_last"] = now

        if elapsed_minutes < 0.01:
            return 0.0

        stats = load_stats(user_id)
        stats["total_time_minutes"] = float(stats.get("total_time_minutes", 0)) + elapsed_minutes
        save_stats(user_id, stats)
        return elapsed_minutes
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Public: event hooks
# ═══════════════════════════════════════════════════════════════════════════

def on_step_complete(user_id: str, step: int, similarity: float = 0.0, lang: str = "en") -> dict:
    """
    Call when a single step is finished.
    Returns {
        "xp_earned": int, "xp_total": int, "level_num": int, "level_name": str,
        "leveled_up": bool, "new_badges": list[tuple]
    }
    """
    tick_time(user_id)   # record elapsed time for this activity interval
    stats    = load_stats(user_id)
    old_xp   = int(stats.get("xp_total", 0))
    old_lvl, *_ = get_level(old_xp)

    xp_earned = STEP_XP.get(step, 5)
    if similarity > 0.80:
        xp_earned += HIGH_ACCURACY_XP

    # Daily XP tracking
    today = date.today().isoformat()
    if stats.get("daily_xp_date") != today:
        stats["daily_xp_date"] = today
        stats["daily_xp"]      = 0
    stats["daily_xp"] = int(stats.get("daily_xp", 0)) + xp_earned
    stats["xp_total"] = old_xp + xp_earned

    new_badges = _check_badges(stats, lang)
    save_stats(user_id, stats)

    new_xp  = int(stats["xp_total"])
    new_lvl, new_name, *_ = get_level(new_xp, lang)

    return {
        "xp_earned":  xp_earned,
        "xp_total":   new_xp,
        "level_num":  new_lvl,
        "level_name": new_name,
        "leveled_up": new_lvl > old_lvl,
        "new_badges": new_badges,
    }


def on_lesson_complete(user_id: str, lang: str = "en") -> dict:
    """
    Call when a whole lesson is finished (after step 8 or explicit completion).
    Returns same shape as on_step_complete plus streak info.
    """
    tick_time(user_id)   # record elapsed time for this activity interval
    stats    = load_stats(user_id)
    old_xp   = int(stats.get("xp_total", 0))
    old_lvl, *_ = get_level(old_xp)

    # Bonus XP
    today = date.today().isoformat()
    if stats.get("daily_xp_date") != today:
        stats["daily_xp_date"] = today
        stats["daily_xp"]      = 0
    stats["daily_xp"]          = int(stats.get("daily_xp", 0)) + LESSON_BONUS_XP
    stats["xp_total"]          = old_xp + LESSON_BONUS_XP
    stats["lessons_completed"] = int(stats.get("lessons_completed", 0)) + 1

    # Streak
    _update_streak(stats)

    new_badges = _check_badges(stats, lang)
    save_stats(user_id, stats)

    new_xp  = int(stats["xp_total"])
    new_lvl, new_name, *_ = get_level(new_xp, lang)

    return {
        "xp_earned":       LESSON_BONUS_XP,
        "xp_total":        new_xp,
        "level_num":       new_lvl,
        "level_name":      new_name,
        "leveled_up":      new_lvl > old_lvl,
        "new_badges":      new_badges,
        "streak_current":  int(stats["streak_current"]),
        "streak_max":      int(stats["streak_max"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public: Streamlit sidebar widget
# ═══════════════════════════════════════════════════════════════════════════

def sidebar_widget(user_id: str) -> None:
    """
    Renders a compact streak / XP / level block into the current sidebar context.
    Language is auto-detected from st.session_state["launcher_native"].
    Call inside a `with st.sidebar:` block (or anywhere in Streamlit).
    """
    try:
        import streamlit as st
        native = st.session_state.get("launcher_native", "English")
        lang   = _NATIVE_TO_CODE.get(native, "en")

        # initialise the session timer on first sidebar render
        import time as _t_mod
        if "_gami_tick_last" not in st.session_state:
            st.session_state["_gami_tick_last"] = _t_mod.monotonic()

        stats  = load_stats(user_id)
        xp     = int(stats.get("xp_total", 0))
        streak = int(stats.get("streak_current", 0))
        total_mins = float(stats.get("total_time_minutes", 0))
        level_num, level_name, pct, xp_to_next = get_level(xp, lang)
        badges  = stats.get("badges", set())
        pct_bar = round(pct * 100)

        streak_color = "#FF6B35" if streak > 0 else "var(--mova-ink-3)"
        if streak == 0:
            streak_label = _t("no_streak", lang)
        elif streak == 1:
            streak_label = f"1 {_t('day', lang)}"
        else:
            streak_label = f"{streak} {_t('days', lang)}"

        bar_html = (
            f'<div style="background:var(--mova-line);border-radius:4px;height:5px;margin:5px 0 3px">'
            f'<div style="background:var(--mova-indigo);width:{pct_bar}%;height:5px;border-radius:4px"></div>'
            f'</div>'
        )
        xp_hint = (_t("xp_next", lang, n=xp_to_next) if xp_to_next > 0
                   else _t("xp_max", lang))

        badge_icons  = "".join(b[1] for b in BADGE_DEFS_RAW if b[0] in badges)[:10]
        time_str     = _format_time(total_mins, lang)
        time_label   = _t("time_label", lang)

        st.markdown(
            f'<div style="background:var(--mova-card);border:1px solid var(--mova-line);'
            f'border-radius:10px;padding:10px 12px;margin-bottom:8px">'
            # row 1: streak + level
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="color:{streak_color};font-size:1rem;font-weight:700">🔥 {streak_label}</span>'
            f'<span style="color:var(--mova-indigo-ink);font-size:.8rem;font-weight:600">'
            f'{_t("level", lang)} {level_num} · {level_name}</span>'
            f'</div>'
            # XP progress bar
            f'{bar_html}'
            # row 2: XP + hint
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="color:var(--mova-ink);font-size:.8rem;font-weight:600">⭐ {xp} XP</span>'
            f'<span style="color:var(--mova-ink-3);font-size:.68rem">{xp_hint}</span>'
            f'</div>'
            # row 3: total study time
            f'<div style="display:flex;align-items:center;margin-top:7px;gap:6px">'
            f'<span style="font-size:.85rem">⏱</span>'
            f'<span style="color:var(--mova-ink-3);font-size:.72rem">{time_label}:</span>'
            f'<span style="color:var(--mova-ink);font-size:.8rem;font-weight:600">{time_str}</span>'
            f'</div>'
            + (f'<div style="margin-top:5px;font-size:.9rem;letter-spacing:2px">{badge_icons}</div>'
               if badge_icons else '')
            + f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass  # never crash the UI for gamification
