"""
engine/vocab_loader.py - Vocabulary loader.

Loads vocabulary.xlsx where each sheet = one topic and each topic
contains many small lessons of ~8 phrases (4 words + 4 example sentences).

The Excel file MUST have these columns per sheet:
    lesson_id   - integer, repeats per topic (1, 1, ..., 2, 2, ..., 3, ...)
    phrase_id   - integer, unique inside the sheet
    en, uk, es, ko - the language columns

Returned DataFrame uses a synthesised global_lesson_id (unique across
the whole workbook) so it slots into the existing 8-step LessonSession
flow without changes.
"""
import pandas as pd

# Same language code mapping as engine/loader.py
LANG_COLUMNS = {
    "English":   "en",
    "Ukrainian": "uk",
    "Spanish":   "es",
    "Korean":    "ko",
}


def _read_all_sheets(db_path: str) -> dict:
    """Return {sheet_name: DataFrame} for every sheet in the workbook."""
    return pd.read_excel(db_path, sheet_name=None, engine="openpyxl")


def _build_global_index(db_path: str):
    """
    Walk the entire workbook and return:
        gid_to_meta: {global_lesson_id: {"topic": str, "local_lesson": int}}
        meta_to_gid: {(topic, local_lesson): global_lesson_id}

    Numbering is stable: assigned in workbook order (sheet order, then
    local lesson_id within the sheet).
    """
    sheets = _read_all_sheets(db_path)
    gid_to_meta = {}
    meta_to_gid = {}
    gid = 0
    for sheet_name, df in sheets.items():
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "lesson_id" not in df.columns:
            # Sheet without lesson_id column -> treat the whole sheet as one lesson
            gid += 1
            gid_to_meta[gid] = {"topic": sheet_name, "local_lesson": 1}
            meta_to_gid[(sheet_name, 1)] = gid
            continue

        # Get unique lesson_ids in this sheet (skip NaN)
        lessons_in_sheet = sorted(
            int(x) for x in df["lesson_id"].dropna().unique()
        )
        for local in lessons_in_sheet:
            gid += 1
            gid_to_meta[gid] = {"topic": sheet_name, "local_lesson": local}
            meta_to_gid[(sheet_name, local)] = gid
    return gid_to_meta, meta_to_gid


def load_vocab(db_path: str, native_lang: str, target_lang: str) -> pd.DataFrame:
    """
    Load vocabulary from Excel and return a DataFrame with columns:
      lesson_id, phrase_id, topic, local_lesson, native, target

    `lesson_id` is the global lesson number (unique across the whole workbook),
    so the same LessonSession plumbing as grammar can be reused.
    `local_lesson` is the lesson number inside the topic (1, 2, 3, ...).

    Rows where either native or target column is empty are dropped, so
    sheets that are not yet translated for the chosen language pair will
    simply yield fewer (or zero) phrases.
    """
    if native_lang not in LANG_COLUMNS or target_lang not in LANG_COLUMNS:
        raise ValueError(
            f"Unsupported language(s): {native_lang} / {target_lang}. "
            f"Supported: {list(LANG_COLUMNS)}"
        )

    native_col = LANG_COLUMNS[native_lang]
    target_col = LANG_COLUMNS[target_lang]

    sheets = _read_all_sheets(db_path)
    _, meta_to_gid = _build_global_index(db_path)

    out_rows = []
    for sheet_name, df in sheets.items():
        df.columns = [str(c).lower().strip() for c in df.columns]

        if native_col not in df.columns or target_col not in df.columns:
            continue

        # Iterate row by row so we keep the explicit lesson_id and phrase_id
        # If the columns are missing, fall back to an implicit numbering.
        has_lesson = "lesson_id" in df.columns
        has_phrase = "phrase_id" in df.columns

        # Drop rows where either language is empty/NaN
        sub = df.copy()
        sub[native_col] = sub[native_col].astype(str).str.strip()
        sub[target_col] = sub[target_col].astype(str).str.strip()
        sub = sub[
            (sub[native_col] != "") &
            (sub[target_col] != "") &
            (sub[native_col].str.lower() != "nan") &
            (sub[target_col].str.lower() != "nan")
        ]

        # Implicit fallback counters when columns are missing
        implicit_phrase = 0
        for _, r in sub.iterrows():
            implicit_phrase += 1
            local_lesson = int(r["lesson_id"]) if has_lesson and pd.notna(r.get("lesson_id")) else 1
            phrase_id    = int(r["phrase_id"]) if has_phrase and pd.notna(r.get("phrase_id")) else implicit_phrase
            gid = meta_to_gid.get((sheet_name, local_lesson))
            if gid is None:
                # local_lesson not found in the global index (shouldn't happen)
                continue
            out_rows.append({
                "lesson_id":    gid,
                "phrase_id":    phrase_id,
                "topic":        sheet_name,
                "local_lesson": local_lesson,
                "native":       r[native_col],
                "target":       r[target_col],
            })

    if not out_rows:
        return pd.DataFrame(
            columns=["lesson_id", "phrase_id", "topic", "local_lesson", "native", "target"]
        )

    result = pd.DataFrame(out_rows)
    result = result.sort_values(["lesson_id", "phrase_id"]).reset_index(drop=True)
    return result


def get_vocab_lesson(df: pd.DataFrame, lesson_id: int) -> pd.DataFrame:
    """Filter DataFrame to a single global lesson_id."""
    return df[df["lesson_id"] == lesson_id].reset_index(drop=True)


def get_available_vocab_lessons(df: pd.DataFrame) -> list:
    """Return sorted list of global lesson_ids that have at least one phrase."""
    if df.empty:
        return []
    return sorted(int(x) for x in df["lesson_id"].unique())


def get_lesson_topics(db_path: str) -> dict:
    """
    Return {global_lesson_id: "Topic - Lesson N"} for every (topic, local_lesson)
    pair found in the workbook. Used by the lesson-picker dropdown.
    """
    gid_to_meta, _ = _build_global_index(db_path)
    return {
        gid: f"{meta['topic']} — Lesson {meta['local_lesson']}"
        for gid, meta in gid_to_meta.items()
    }


def get_topic_for_lesson(db_path: str, global_lesson_id: int) -> tuple:
    """Return (topic, local_lesson) for a global lesson id, or (None, None)."""
    gid_to_meta, _ = _build_global_index(db_path)
    meta = gid_to_meta.get(global_lesson_id)
    if meta is None:
        return (None, None)
    return (meta["topic"], meta["local_lesson"])
