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
import streamlit as st

# Same language code mapping as engine/loader.py
LANG_COLUMNS = {
    "English":   "en",
    "Ukrainian": "uk",
    "Spanish":   "es",
    "Korean":    "ko",
}


@st.cache_data(show_spinner=False)
def _read_all_sheets(db_path: str) -> dict:
    """Return {sheet_name: DataFrame} for every sheet in the workbook.
    Cached so the file is read from disk only once per session."""
    return pd.read_excel(db_path, sheet_name=None, engine="openpyxl")


@st.cache_data(show_spinner=False)
def _build_global_index(db_path: str):
    """
    Walk the entire workbook and return:
        gid_to_meta: {global_lesson_id: {"topic": str, "local_lesson": int}}
        meta_to_gid: {(topic, local_lesson): global_lesson_id}

    Numbering is stable: assigned in workbook order (sheet order, then
    local lesson_id within the sheet).
    Cached so the index is built only once per session.
    """
    sheets = _read_all_sheets(db_path)
    gid_to_meta = {}
    meta_to_gid = {}
    gid = 0
    for sheet_name, df in sheets.items():
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "lesson_id" not in df.columns:
            gid += 1
            gid_to_meta[gid] = {"topic": sheet_name, "local_lesson": 1}
            meta_to_gid[(sheet_name, 1)] = gid
            continue

        lessons_in_sheet = sorted(
            int(x) for x in df["lesson_id"].dropna().unique()
        )
        for local in lessons_in_sheet:
            gid += 1
            gid_to_meta[gid] = {"topic": sheet_name, "local_lesson": local}
            meta_to_gid[(sheet_name, local)] = gid
    return gid_to_meta, meta_to_gid


@st.cache_data(show_spinner=False)
def load_vocab(db_path: str, native_lang: str, target_lang: str) -> pd.DataFrame:
    """
    Load vocabulary from Excel and return a DataFrame with columns:
      lesson_id, phrase_id, topic, local_lesson, native, target

    `lesson_id` is the global lesson number (unique across the whole workbook).
    `local_lesson` is the lesson number inside the topic (1, 2, 3, ...).

    Rows where either native or target column is empty are dropped.
    Cached: the Excel file is not re-read on every Streamlit rerun.
    """
    if native_lang not in LANG_COLUMNS or target_lang not in LANG_COLUMNS:
        raise ValueError(
            f"Unsupported language(s): {native_lang} / {target_lang}. "
            f"Supported: {list(LANG_COLUMNS)}"
        )

    native_col = LANG_COLUMNS[native_lang]
    target_col = LANG_COLUMNS[target_lang]

    # Reuse cached reads — no double file I/O
    sheets = _read_all_sheets(db_path)
    _, meta_to_gid = _build_global_index(db_path)

    chunks = []
    for sheet_name, df in sheets.items():
        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]

        if native_col not in df.columns or target_col not in df.columns:
            continue

        has_lesson = "lesson_id" in df.columns
        has_phrase = "phrase_id" in df.columns

        # Vectorised string cleaning (no iterrows)
        df[native_col] = df[native_col].astype(str).str.strip()
        df[target_col] = df[target_col].astype(str).str.strip()
        mask = (
            (df[native_col] != "") & (df[native_col].str.lower() != "nan") &
            (df[target_col] != "") & (df[target_col].str.lower() != "nan")
        )
        sub = df[mask].copy()

        if sub.empty:
            continue

        # Local lesson column
        if has_lesson:
            sub["_local_lesson"] = sub["lesson_id"].fillna(1).astype(int)
        else:
            sub["_local_lesson"] = 1

        # Phrase id column
        if has_phrase:
            sub["_phrase_id"] = (
                sub["phrase_id"]
                .where(sub["phrase_id"].notna(), other=range(1, len(sub) + 1))
                .astype(int)
            )
        else:
            sub["_phrase_id"] = range(1, len(sub) + 1)

        # Map (sheet, local_lesson) → global lesson id
        sub["_gid"] = sub["_local_lesson"].map(
            lambda local, sn=sheet_name: meta_to_gid.get((sn, local))
        )
        sub = sub.dropna(subset=["_gid"])
        if sub.empty:
            continue

        chunk = pd.DataFrame({
            "lesson_id":    sub["_gid"].astype(int),
            "phrase_id":    sub["_phrase_id"].values,
            "topic":        sheet_name,
            "local_lesson": sub["_local_lesson"].values,
            "native":       sub[native_col].values,
            "target":       sub[target_col].values,
        })
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            columns=["lesson_id", "phrase_id", "topic", "local_lesson", "native", "target"]
        )

    result = pd.concat(chunks, ignore_index=True)
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def get_topic_for_lesson(db_path: str, global_lesson_id: int) -> tuple:
    """Return (topic, local_lesson) for a global lesson id, or (None, None)."""
    gid_to_meta, _ = _build_global_index(db_path)
    meta = gid_to_meta.get(global_lesson_id)
    if meta is None:
        return (None, None)
    return (meta["topic"], meta["local_lesson"])


@st.cache_data(show_spinner=False)
def get_vocab_nav_data(db_path: str) -> dict:
    """
    Return structured navigation data for the hierarchical vocab picker.

    Returns:
        {
            sheet_name: [
                {"gid": int, "local_lesson": int, "name": str},
                ...
            ]
        }

    `name` comes from the `lesson_name` column if it exists in the workbook,
    otherwise falls back to "Lesson N".
    The list is sorted by local_lesson (ascending).
    """
    # Both calls hit the cache — no extra file reads
    sheets = _read_all_sheets(db_path)
    gid_to_meta, meta_to_gid = _build_global_index(db_path)

    result: dict = {}
    for sheet_name, df in sheets.items():
        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        has_lesson = "lesson_id" in df.columns
        has_name   = "lesson_name" in df.columns

        if not has_lesson:
            gid = meta_to_gid.get((sheet_name, 1))
            if gid:
                result[sheet_name] = [
                    {"gid": gid, "local_lesson": 1, "name": sheet_name}
                ]
            continue

        # Build lesson_id → name map vectorially
        lesson_names: dict = {}
        if has_name:
            # Take first non-empty lesson_name per lesson_id
            name_df = (
                df[["lesson_id", "lesson_name"]]
                .dropna(subset=["lesson_id"])
                .copy()
            )
            name_df["lesson_id"] = name_df["lesson_id"].astype(int)
            name_df["lesson_name"] = name_df["lesson_name"].astype(str).str.strip()
            name_df = name_df[name_df["lesson_name"].str.lower() != "nan"]
            name_df = name_df[name_df["lesson_name"] != ""]
            for lid, name in name_df.groupby("lesson_id")["lesson_name"].first().items():
                lesson_names[int(lid)] = name

        # Fill missing with "Lesson N"
        all_lids = sorted(
            int(x) for x in df["lesson_id"].dropna().unique()
        )
        for lid in all_lids:
            lesson_names.setdefault(lid, f"Lesson {lid}")

        lessons = []
        for local_lid in sorted(lesson_names.keys()):
            gid = meta_to_gid.get((sheet_name, local_lid))
            if gid is None:
                continue
            lessons.append({
                "gid":          gid,
                "local_lesson": local_lid,
                "name":         lesson_names[local_lid],
            })
        if lessons:
            result[sheet_name] = lessons

    return result
