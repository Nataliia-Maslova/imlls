"""
engine/vocab_loader.py - Vocabulary loader with lazy per-sheet loading.

Loads vocabulary.xlsx where each sheet = one topic and each topic
contains many small lessons of ~8 phrases (4 words + 4 example sentences).

The Excel file MUST have these columns per sheet:
    lesson_id   - integer, repeats per topic (1, 1, ..., 2, 2, ..., 3, ...)
    phrase_id   - integer, unique inside the sheet
    en, uk, es, ko - the language columns

Returned DataFrame uses a synthesised global_lesson_id (unique across
the whole workbook) so it slots into the existing 8-step LessonSession
flow without changes.

Lazy loading strategy
---------------------
* _read_sheet_index()  -- reads only lesson_id + lesson_name columns from
  every sheet; lightweight, cached, used for navigation & global index.
* _read_single_sheet() -- reads one full sheet on demand, cached per sheet;
  only triggered when the user actually opens a topic/lesson.
* load_vocab()         -- accepts optional `topic` kwarg; when provided,
  only the requested sheet is loaded (fast). When None, all sheets are
  loaded via the per-sheet cache (backward-compatible).
"""
import pandas as pd
import streamlit as st

# Same language code mapping as engine/loader.py
LANG_COLUMNS = {
    "English":    "en",
    "Ukrainian":  "uk",
    "Spanish":    "es",
    "Korean":     "ko",
    "French":     "fr",
    "German":     "de",
    "Japanese":   "ja",
    "Chinese":    "zh",
    "Portuguese": "pt",
    "Italian":    "it",
    "Polish":     "pl",
    "Russian":    "ru",
}


# --- Lightweight index (sheet names + lesson_id/lesson_name only) ----------

@st.cache_data(show_spinner=False)
def _read_sheet_index(db_path: str) -> dict:
    """Read only lesson_id and lesson_name columns from every sheet.

    Returns {sheet_name: DataFrame} where each DataFrame has at most
    two columns: lesson_id, lesson_name (whichever exist in the sheet).
    This is ~10-20x faster than reading all phrase text columns.
    Cached so the file is touched only once per session.
    """
    xl = pd.ExcelFile(db_path, engine="openpyxl")
    result = {}
    for sheet_name in xl.sheet_names:
        try:
            df = xl.parse(sheet_name)
            df.columns = [str(c).lower().strip() for c in df.columns]
            keep = [c for c in ("lesson_id", "lesson_name") if c in df.columns]
            result[sheet_name] = df[keep] if keep else pd.DataFrame()
        except Exception:
            result[sheet_name] = pd.DataFrame()
    return result


# --- Per-sheet full data loader --------------------------------------------

@st.cache_data(show_spinner=False)
def _read_single_sheet(db_path: str, sheet_name: str) -> pd.DataFrame:
    """Read all columns for one sheet. Cached per (db_path, sheet_name).

    The cache means that once a topic is first opened, every subsequent
    visit in the same session is instant.
    """
    df = pd.read_excel(db_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


# --- Global lesson ID index ------------------------------------------------

@st.cache_data(show_spinner=False)
def _build_global_index(db_path: str):
    """
    Walk the workbook (index only -- no phrase text) and return:
        gid_to_meta: {global_lesson_id: {"topic": str, "local_lesson": int}}
        meta_to_gid: {(topic, local_lesson): global_lesson_id}

    Numbering is stable: assigned in workbook order (sheet order, then
    local lesson_id within the sheet).
    Cached so the index is built only once per session.
    """
    index = _read_sheet_index(db_path)
    gid_to_meta = {}
    meta_to_gid = {}
    gid = 0
    for sheet_name, df in index.items():
        if "lesson_id" not in df.columns or df.empty:
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


# --- Helpers ---------------------------------------------------------------

def _process_sheet(sheet_name, df, native_col, target_col, meta_to_gid):
    """Convert one raw sheet DataFrame into the canonical vocab format.

    Returns a DataFrame or None if the sheet has no usable phrases.
    """
    df = df.copy()
    if native_col not in df.columns or target_col not in df.columns:
        return None

    has_lesson = "lesson_id" in df.columns
    has_phrase = "phrase_id" in df.columns

    # Vectorised string cleaning
    df[native_col] = df[native_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()
    mask = (
        (df[native_col] != "") & (df[native_col].str.lower() != "nan") &
        (df[target_col] != "") & (df[target_col].str.lower() != "nan")
    )
    sub = df[mask].copy()
    if sub.empty:
        return None

    sub["_local_lesson"] = sub["lesson_id"].fillna(1).astype(int) if has_lesson else 1

    if has_phrase:
        sub["_phrase_id"] = (
            sub["phrase_id"]
            .where(sub["phrase_id"].notna(), other=range(1, len(sub) + 1))
            .astype(int)
        )
    else:
        sub["_phrase_id"] = range(1, len(sub) + 1)

    sub["_gid"] = sub["_local_lesson"].map(
        lambda local, sn=sheet_name: meta_to_gid.get((sn, local))
    )
    sub = sub.dropna(subset=["_gid"])
    if sub.empty:
        return None

    return pd.DataFrame({
        "lesson_id":    sub["_gid"].astype(int),
        "phrase_id":    sub["_phrase_id"].values,
        "topic":        sheet_name,
        "local_lesson": sub["_local_lesson"].values,
        "native":       sub[native_col].values,
        "target":       sub[target_col].values,
    })


# --- Public API ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_vocab(db_path, native_lang, target_lang, topic=None):
    """
    Load vocabulary and return a DataFrame with columns:
      lesson_id, phrase_id, topic, local_lesson, native, target

    `lesson_id` is the global lesson number (unique across the whole workbook).
    `local_lesson` is the lesson number inside the topic (1, 2, 3, ...).

    Parameters
    ----------
    topic : str | None
        When given, only the phrases for that sheet/topic are loaded --
        much faster than loading the whole workbook. Pass None (default)
        to load all topics (backward-compatible).

    Rows where either native or target column is empty are dropped.
    Results are cached per (db_path, native_lang, target_lang, topic).
    """
    if native_lang not in LANG_COLUMNS or target_lang not in LANG_COLUMNS:
        raise ValueError(
            "Unsupported language(s): {} / {}. Supported: {}".format(
                native_lang, target_lang, list(LANG_COLUMNS))
        )

    native_col = LANG_COLUMNS[native_lang]
    target_col = LANG_COLUMNS[target_lang]

    _, meta_to_gid = _build_global_index(db_path)

    if topic is not None:
        # Lazy path: load one sheet only
        df = _read_single_sheet(db_path, topic)
        chunk = _process_sheet(topic, df, native_col, target_col, meta_to_gid)
        if chunk is None:
            return pd.DataFrame(
                columns=["lesson_id", "phrase_id", "topic",
                         "local_lesson", "native", "target"]
            )
        return chunk.sort_values(["lesson_id", "phrase_id"]).reset_index(drop=True)

    # Full load path: iterate sheets via per-sheet cache
    index = _read_sheet_index(db_path)
    chunks = []
    for sheet_name in index:
        df = _read_single_sheet(db_path, sheet_name)
        chunk = _process_sheet(sheet_name, df, native_col, target_col, meta_to_gid)
        if chunk is not None:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            columns=["lesson_id", "phrase_id", "topic", "local_lesson", "native", "target"]
        )

    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values(["lesson_id", "phrase_id"]).reset_index(drop=True)
    return result


def get_vocab_lesson(df, lesson_id):
    """Filter DataFrame to a single global lesson_id."""
    return df[df["lesson_id"] == lesson_id].reset_index(drop=True)


def get_available_vocab_lessons(df):
    """Return sorted list of global lesson_ids that have at least one phrase."""
    if df.empty:
        return []
    return sorted(int(x) for x in df["lesson_id"].unique())


@st.cache_data(show_spinner=False)
def get_lesson_topics(db_path):
    """
    Return {global_lesson_id: "Topic - Lesson N"} for every (topic, local_lesson)
    pair found in the workbook. Used by the lesson-picker dropdown.
    Built from the lightweight index -- no phrase data loaded.
    """
    gid_to_meta, _ = _build_global_index(db_path)
    return {
        gid: "{} — Lesson {}".format(meta["topic"], meta["local_lesson"])
        for gid, meta in gid_to_meta.items()
    }


@st.cache_data(show_spinner=False)
def get_topic_for_lesson(db_path, global_lesson_id):
    """Return (topic, local_lesson) for a global lesson id, or (None, None).
    Built from the lightweight index -- no phrase data loaded.
    """
    gid_to_meta, _ = _build_global_index(db_path)
    meta = gid_to_meta.get(global_lesson_id)
    if meta is None:
        return (None, None)
    return (meta["topic"], meta["local_lesson"])


@st.cache_data(show_spinner=False)
def get_vocab_nav_data(db_path):
    """
    Return structured navigation data for the hierarchical vocab picker.
    Built from the lightweight index -- no phrase data loaded.

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
    index = _read_sheet_index(db_path)
    gid_to_meta, meta_to_gid = _build_global_index(db_path)

    result = {}
    for sheet_name, df in index.items():
        has_lesson = "lesson_id" in df.columns
        has_name   = "lesson_name" in df.columns

        if not has_lesson or df.empty:
            gid = meta_to_gid.get((sheet_name, 1))
            if gid:
                result[sheet_name] = [
                    {"gid": gid, "local_lesson": 1, "name": sheet_name}
                ]
            continue

        # Build lesson_id -> name map vectorially (index data only)
        lesson_names = {}
        if has_name:
            name_df = df[["lesson_id", "lesson_name"]].dropna(subset=["lesson_id"]).copy()
            name_df["lesson_id"] = name_df["lesson_id"].astype(int)
            name_df["lesson_name"] = name_df["lesson_name"].astype(str).str.strip()
            name_df = name_df[name_df["lesson_name"].str.lower() != "nan"]
            name_df = name_df[name_df["lesson_name"] != ""]
            for lid, name in name_df.groupby("lesson_id")["lesson_name"].first().items():
                lesson_names[int(lid)] = name

        all_lids = sorted(int(x) for x in df["lesson_id"].dropna().unique())
        for lid in all_lids:
            lesson_names.setdefault(lid, "Lesson {}".format(lid))

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
