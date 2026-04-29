"""
Session logger — writes every user interaction to:
  1. Local CSV file (always, fast)
  2. Google Sheets (when credentials available — Streamlit Cloud or local with secrets)
"""
import csv
import os
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "timestamp", "user_id", "lesson_id", "phrase_id",
    "step", "similarity", "response_time_ms",
    "attempts", "success", "mode",
]

SHEET_NAME = "IMLLS_Logs"   # must match your Google Sheet name exactly


# ── Google Sheets connection (lazy, singleton) ────────────────────────────

_gsheet = None   # cache the worksheet object


def _get_worksheet():
    """
    Returns the gspread Worksheet object, or None if credentials not available.
    Credentials are read from st.secrets (Streamlit Cloud) or from a local
    secrets.toml file at .streamlit/secrets.toml.
    """
    global _gsheet
    if _gsheet is not None:
        return _gsheet

    try:
        import gspread
        from google.oauth2.service_account import Credentials
        import streamlit as st

        # Read credentials from st.secrets
        creds_dict = dict(st.secrets["gcp_service_account"])

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)

        # Open sheet and get or create the right worksheet tab
        spreadsheet = client.open(SHEET_NAME)

        # Try to get existing worksheet named "logs", create if missing
        try:
            worksheet = spreadsheet.worksheet("logs")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title="logs", rows=10000, cols=len(COLUMNS))
            # Write header row
            worksheet.append_row(COLUMNS, value_input_option="RAW")

        _gsheet = worksheet
        return _gsheet

    except Exception as e:
        # Silently fail — CSV logging still works
        print(f"[Logger] Google Sheets not available: {e}")
        return None


def _append_to_sheet(row: dict):
    """Append one row to Google Sheets. Non-blocking on failure."""
    try:
        ws = _get_worksheet()
        if ws is None:
            return
        values = [str(row[col]) for col in COLUMNS]
        ws.append_row(values, value_input_option="RAW")
    except Exception as e:
        print(f"[Logger] Sheet append error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SessionLogger
# ═══════════════════════════════════════════════════════════════════════════

class SessionLogger:
    def __init__(self, user_id: str):
        self.user_id  = user_id
        self.log_path = LOGS_DIR / f"{user_id}.csv"
        self._ensure_header()

    def _ensure_header(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()

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
            "lesson_id":        lesson_id,
            "phrase_id":        phrase_id,
            "step":             step,
            "similarity":       round(similarity, 4),
            "response_time_ms": response_time_ms,
            "attempts":         attempts,
            "success":          int(success),
            "mode":             mode,
        }

        # 1. Always write to local CSV
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)

        # 2. Also write to Google Sheets (fails silently if not configured)
        _append_to_sheet(row)

    def count(self) -> int:
        if not self.log_path.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1

    def read_all(self) -> list[dict]:
        """Return list of dicts from local CSV."""
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
