"""
Session logger — writes every user interaction to:
  1. Local CSV file (always, fast)
  2. Google Sheets worksheet "logs" (interaction log)
  3. Google Sheets worksheet "progress" (last completed lesson per user+lang_pair)
"""
import csv
from datetime import datetime
from pathlib import Path

LOGS_DIR   = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
SHEET_NAME = "IMLLS_Logs"

COLUMNS = [
    "timestamp", "user_id", "language_pair", "lesson_id", "phrase_id",
    "step", "similarity", "response_time_ms", "attempts", "success", "mode",
]

# ── Google Sheets — lazy singletons ──────────────────────────────────────

_spreadsheet   = None   # gspread Spreadsheet object
_ws_logs       = None   # worksheet "logs"
_ws_progress   = None   # worksheet "progress"


def _get_spreadsheet():
    global _spreadsheet
    if _spreadsheet is not None:
        return _spreadsheet
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        import streamlit as st

        creds_dict = dict(st.secrets["gcp_service_account"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        _spreadsheet = client.open(SHEET_NAME)
        return _spreadsheet
    except Exception as e:
        print(f"[Logger] Google Sheets unavailable: {e}")
        return None


def _get_ws_logs():
    """Get or create the 'logs' worksheet."""
    global _ws_logs
    if _ws_logs is not None:
        return _ws_logs
    try:
        import gspread
        ss = _get_spreadsheet()
        if ss is None:
            return None
        try:
            _ws_logs = ss.worksheet("logs")
        except gspread.exceptions.WorksheetNotFound:
            _ws_logs = ss.add_worksheet(title="logs", rows=50000, cols=len(COLUMNS))
            _ws_logs.append_row(COLUMNS, value_input_option="RAW")
        return _ws_logs
    except Exception as e:
        print(f"[Logger] logs worksheet error: {e}")
        return None


def _get_ws_progress():
    """Get or create the 'progress' worksheet."""
    global _ws_progress
    if _ws_progress is not None:
        return _ws_progress
    try:
        import gspread
        ss = _get_spreadsheet()
        if ss is None:
            return None
        progress_cols = [
            "user_id", "language_pair",
            "last_completed_lesson", "last_step", "updated_at"
        ]
        try:
            _ws_progress = ss.worksheet("progress")
        except gspread.exceptions.WorksheetNotFound:
            _ws_progress = ss.add_worksheet(title="progress", rows=1000, cols=len(progress_cols))
            _ws_progress.append_row(progress_cols, value_input_option="RAW")
        return _ws_progress
    except Exception as e:
        print(f"[Logger] progress worksheet error: {e}")
        return None


# ── Progress helpers ──────────────────────────────────────────────────────

def get_last_lesson(user_id: str, language_pair: str) -> int | None:
    """
    Returns the last completed lesson number for this user+language_pair,
    or None if no progress found.
    """
    try:
        ws = _get_ws_progress()
        if ws is None:
            return None
        records = ws.get_all_records()
        for row in records:
            if (str(row.get("user_id")) == user_id and
                    str(row.get("language_pair")) == language_pair):
                val = row.get("last_completed_lesson")
                return int(val) if val else None
        return None
    except Exception as e:
        print(f"[Logger] get_last_lesson error: {e}")
        return None


def save_progress(user_id: str, language_pair: str,
                  last_completed_lesson: int, last_step: int):
    """
    Upsert progress row for user+language_pair.
    If row exists — update it. If not — append new row.
    """
    try:
        ws = _get_ws_progress()
        if ws is None:
            return
        records  = ws.get_all_records()
        now      = datetime.now().isoformat()
        new_vals = [user_id, language_pair,
                    last_completed_lesson, last_step, now]

        # Find existing row index (1-based, +1 for header)
        for i, row in enumerate(records, start=2):
            if (str(row.get("user_id")) == user_id and
                    str(row.get("language_pair")) == language_pair):
                # Update existing row
                ws.update(f"A{i}:E{i}", [new_vals])
                return

        # No existing row — append
        ws.append_row(new_vals, value_input_option="RAW")

    except Exception as e:
        print(f"[Logger] save_progress error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SessionLogger
# ═══════════════════════════════════════════════════════════════════════════

class SessionLogger:
    def __init__(self, user_id: str, language_pair: str = ""):
        self.user_id       = user_id
        self.language_pair = language_pair          # e.g. "en-uk"
        self.log_path      = LOGS_DIR / f"{user_id}.csv"
        self._ensure_header()

    def _ensure_header(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def log(
        self,
        lesson_id:        int,
        phrase_id:        int,
        step:             int,
        similarity:       float,
        response_time_ms: int,
        attempts:         int,
        success:          bool,
        mode:             str,
    ):
        row = {
            "timestamp":        datetime.now().isoformat(),
            "user_id":          self.user_id,
            "language_pair":    self.language_pair,
            "lesson_id":        lesson_id,
            "phrase_id":        phrase_id,
            "step":             step,
            "similarity":       round(similarity, 4),
            "response_time_ms": response_time_ms,
            "attempts":         attempts,
            "success":          int(success),
            "mode":             mode,
        }

        # 1. Local CSV
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)

        # 2. Google Sheets logs tab
        try:
            ws = _get_ws_logs()
            if ws:
                ws.append_row([str(row[c]) for c in COLUMNS],
                              value_input_option="RAW")
        except Exception as e:
            print(f"[Logger] log append error: {e}")

    def complete_lesson(self, lesson_id: int, step: int = 7):
        """Call when a lesson is fully completed to update progress."""
        save_progress(self.user_id, self.language_pair, lesson_id, step)

    def count(self) -> int:
        if not self.log_path.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
