"""
Custom phrases storage — user-created lessons for the same 8-step flow.

Local CSV is the source of truth (data/custom_phrases.csv). If a
Google Sheets connection is available (same `gcp_service_account`
secrets used by SessionLogger), changes are mirrored to a worksheet
called ``custom_phrases`` so data survives Streamlit Cloud redeploys.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR  = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH  = DATA_DIR / "custom_phrases.csv"
SHEET_TAB = "custom_phrases"

COLUMNS = [
    "user_id", "lesson_id", "lesson_name", "phrase_id",
    "native_lang", "target_lang",
    "native", "target",
    "created_at",
]


# ─── Local CSV ────────────────────────────────────────────────────────────

def _ensure_csv() -> None:
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def _read_all() -> pd.DataFrame:
    _ensure_csv()
    try:
        df = pd.read_csv(CSV_PATH, dtype={"lesson_id": "Int64", "phrase_id": "Int64"})
    except Exception:
        df = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df[COLUMNS]


def _write_all(df: pd.DataFrame) -> None:
    df = df[COLUMNS].copy()
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")


# ─── Google Sheets mirror (optional, silent failure) ──────────────────────

_ws_custom = None


def _get_ws():
    """Return the gspread worksheet for the custom_phrases tab, or None."""
    global _ws_custom
    if _ws_custom is not None:
        return _ws_custom
    try:
        # Reuse the same auth path as engine.logger
        from engine.logger import _get_spreadsheet
        import gspread
        ss = _get_spreadsheet()
        if ss is None:
            return None
        try:
            _ws_custom = ss.worksheet(SHEET_TAB)
        except gspread.exceptions.WorksheetNotFound:
            _ws_custom = ss.add_worksheet(title=SHEET_TAB, rows=5000,
                                           cols=len(COLUMNS))
            _ws_custom.append_row(COLUMNS, value_input_option="RAW")
        return _ws_custom
    except Exception as e:
        print(f"[custom_store] Google Sheets unavailable: {e}")
        return None


def _sync_to_sheets(df: pd.DataFrame) -> None:
    """Replace the worksheet contents with the given DataFrame. Silent on failure."""
    try:
        ws = _get_ws()
        if ws is None:
            return
        ws.clear()
        ws.append_row(COLUMNS, value_input_option="RAW")
        rows = df[COLUMNS].astype(str).values.tolist()
        if rows:
            ws.append_rows(rows, value_input_option="RAW")
    except Exception as e:
        print(f"[custom_store] sheet sync failed: {e}")


def _pull_from_sheets() -> pd.DataFrame | None:
    """Pull the worksheet contents. Returns DataFrame or None if unavailable."""
    try:
        ws = _get_ws()
        if ws is None:
            return None
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(records)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df["lesson_id"] = pd.to_numeric(df["lesson_id"], errors="coerce").astype("Int64")
        df["phrase_id"] = pd.to_numeric(df["phrase_id"], errors="coerce").astype("Int64")
        return df[COLUMNS]
    except Exception as e:
        print(f"[custom_store] sheet pull failed: {e}")
        return None


# ─── Public API ──────────────────────────────────────────────────────────

_sheets_synced: bool = False   # pulled from Sheets at most once per process start


def load_all(user_id: str | None = None, prefer_sheets: bool = True) -> pd.DataFrame:
    """Load every custom phrase for the user (or all users if user_id is None).

    When ``prefer_sheets`` is True and the sheet has more rows than the local
    CSV, the local CSV is refreshed from Sheets — but only ONCE per process
    start (not on every rerun) to avoid repeated slow API calls.
    This handles Streamlit Cloud redeploys that wipe the local filesystem.
    """
    global _sheets_synced
    local = _read_all()
    if prefer_sheets and not _sheets_synced:
        remote = _pull_from_sheets()
        if remote is not None and len(remote) > len(local):
            _write_all(remote)
            local = remote
        _sheets_synced = True
    if user_id:
        local = local[local["user_id"].astype(str) == str(user_id)]
    return local.reset_index(drop=True)


def list_user_lessons(user_id: str,
                       native_lang: str | None = None,
                       target_lang: str | None = None) -> pd.DataFrame:
    """Return a DataFrame [lesson_id, lesson_name, phrases, native_lang, target_lang]
    grouped per lesson, filtered to the user (and language pair if given)."""
    df = load_all(user_id=user_id)
    if df.empty:
        return pd.DataFrame(columns=["lesson_id", "lesson_name", "phrases",
                                     "native_lang", "target_lang"])
    if native_lang:
        df = df[df["native_lang"] == native_lang]
    if target_lang:
        df = df[df["target_lang"] == target_lang]
    if df.empty:
        return pd.DataFrame(columns=["lesson_id", "lesson_name", "phrases",
                                     "native_lang", "target_lang"])
    out = (df.groupby(["lesson_id", "lesson_name",
                        "native_lang", "target_lang"], dropna=False)
             .size()
             .reset_index(name="phrases")
             .sort_values("lesson_id"))
    return out.reset_index(drop=True)


def get_lesson_phrases(user_id: str, lesson_id: int) -> pd.DataFrame:
    """Return phrases for one user lesson in the format the 8-step flow expects.

    Output columns: lesson_id, phrase_id, difficulty, native, target.
    """
    df = load_all(user_id=user_id)
    df = df[df["lesson_id"].astype("Int64") == int(lesson_id)]
    if df.empty:
        return pd.DataFrame(columns=["lesson_id", "phrase_id", "difficulty",
                                     "native", "target"])
    df = df.sort_values("phrase_id").reset_index(drop=True)
    df["difficulty"] = 1  # neutral default
    return df[["lesson_id", "phrase_id", "difficulty", "native", "target"]]


def _next_lesson_id(user_id: str) -> int:
    df = load_all(user_id=user_id, prefer_sheets=False)
    if df.empty:
        return 1
    try:
        return int(df["lesson_id"].dropna().astype(int).max()) + 1
    except Exception:
        return 1


def add_lesson(user_id: str,
                lesson_name: str,
                native_lang: str,
                target_lang: str,
                pairs: Iterable[tuple[str, str]]) -> int:
    """Create a new lesson with the given native↔target phrase pairs.

    Returns the assigned lesson_id. Empty pairs are filtered out.
    """
    cleaned = [(n.strip(), t.strip()) for n, t in pairs
               if n and t and n.strip() and t.strip()]
    if not cleaned:
        raise ValueError("Потрібна щонайменше одна валідна пара фраз")
    if not lesson_name.strip():
        lesson_name = f"My lesson {datetime.now():%Y-%m-%d %H:%M}"

    lesson_id = _next_lesson_id(user_id)
    now = datetime.now().isoformat(timespec="seconds")

    new_rows = []
    for i, (nat, tgt) in enumerate(cleaned, start=1):
        new_rows.append({
            "user_id":     user_id,
            "lesson_id":   lesson_id,
            "lesson_name": lesson_name.strip(),
            "phrase_id":   i,
            "native_lang": native_lang,
            "target_lang": target_lang,
            "native":      nat,
            "target":      tgt,
            "created_at":  now,
        })

    all_df = _read_all()
    all_df = pd.concat([all_df, pd.DataFrame(new_rows)], ignore_index=True)
    _write_all(all_df)
    _sync_to_sheets(all_df)
    return lesson_id


def delete_lesson(user_id: str, lesson_id: int) -> bool:
    """Remove all phrases of a given user lesson."""
    df = _read_all()
    mask = (df["user_id"].astype(str) == str(user_id)) & \
           (df["lesson_id"].astype("Int64") == int(lesson_id))
    if not mask.any():
        return False
    df = df[~mask]
    _write_all(df)
    _sync_to_sheets(df)
    return True


def rename_lesson(user_id: str, lesson_id: int, new_name: str) -> bool:
    """Change the display name of a user lesson."""
    new_name = new_name.strip()
    if not new_name:
        return False
    df = _read_all()
    mask = (df["user_id"].astype(str) == str(user_id)) & \
           (df["lesson_id"].astype("Int64") == int(lesson_id))
    if not mask.any():
        return False
    df.loc[mask, "lesson_name"] = new_name
    _write_all(df)
    _sync_to_sheets(df)
    return True


def parse_pairs_text(text: str, sep: str = "=") -> list[tuple[str, str]]:
    """Parse a chunk of text in the form

        native = target
        another native = another target
        ...

    Returns a list of (native, target) tuples. Lines without ``sep`` are
    skipped silently.
    """
    pairs = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or sep not in line:
            continue
        nat, _, tgt = line.partition(sep)
        nat, tgt = nat.strip(), tgt.strip()
        if nat and tgt:
            pairs.append((nat, tgt))
    return pairs
