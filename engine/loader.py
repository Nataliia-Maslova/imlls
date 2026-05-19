"""
Data loader — reads the Excel database and returns filtered DataFrames.
"""
import pandas as pd
from pathlib import Path

# Language column names in the Excel file
LANG_COLUMNS = {
    "English":    "en",
    "Ukrainian":  "uk",
    "Spanish":    "es",
    "Korean":     "ko",
}

# Whisper language codes
WHISPER_LANG = {
    "English":   "en",
    "Ukrainian": "uk",
    "Spanish":   "es",
    "Korean":    "ko",
}

# gTTS language codes
TTS_LANG = {
    "English":   "en",
    "Ukrainian": "uk",
    "Spanish":   "es",
    "Korean":    "ko",
}


def load_phrases(db_path: str, native_lang: str, target_lang: str) -> pd.DataFrame:
    """
    Load phrases from Excel and return a DataFrame with columns:
    lesson_id, phrase_id, difficulty, native, target,
    plus optional topic_en / topic_uk (if present in the source).
    """
    df = pd.read_excel(db_path, sheet_name="phrases")

    # Normalize column names (lowercase)
    df.columns = [c.lower().strip() for c in df.columns]

    native_col  = LANG_COLUMNS[native_lang].lower()
    target_col  = LANG_COLUMNS[target_lang].lower()

    # Keep only rows where both columns are filled and not TODO
    mask = (
        df[native_col].notna() &
        df[target_col].notna() &
        (df[native_col].astype(str).str.strip() != "TODO") &
        (df[target_col].astype(str).str.strip() != "TODO")
    )
    df = df[mask].copy()

    keep = ["lesson_id", "phrase_id", "difficulty", native_col, target_col]
    for opt in ("topic_en", "topic_uk"):
        if opt in df.columns:
            keep.append(opt)
    result = df[keep].copy()
    result = result.rename(columns={native_col: "native", target_col: "target"})
    result = result.sort_values(["lesson_id", "phrase_id"]).reset_index(drop=True)
    return result


def get_lesson(df: pd.DataFrame, lesson_id: int) -> pd.DataFrame:
    return df[df["lesson_id"] == lesson_id].reset_index(drop=True)


def get_available_lessons(df: pd.DataFrame) -> list[int]:
    return sorted(df["lesson_id"].unique().tolist())


def get_lesson_topics(df: pd.DataFrame, native_lang: str = "English") -> dict:
    """
    Return {lesson_id: topic_str} using the column matching the user's
    native language (topic_uk for Ukrainians, topic_en otherwise).
    If the topic columns aren't present in the workbook, returns {}.
    """
    # Prefer the topic in the user's native language; fall back to English
    if native_lang == "Ukrainian" and "topic_uk" in df.columns:
        col = "topic_uk"
    elif "topic_en" in df.columns:
        col = "topic_en"
    else:
        return {}
    out = {}
    for lid, sub in df.groupby("lesson_id"):
        val = sub[col].dropna().astype(str).str.strip()
        if len(val):
            out[int(lid)] = val.iloc[0]
    return out
