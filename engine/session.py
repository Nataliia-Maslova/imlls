"""
LessonSession — state machine for lesson-based learning (7 steps).
"""
import time
from dataclasses import dataclass, field
from engine.scorer import evaluate
from engine.logger import SessionLogger


@dataclass
class SessionState:
    user_id:       str
    lesson_id:     int
    native_lang:   str
    target_lang:   str
    current_step:  int   = 1
    phrase_index:  int   = 0
    step_start_ts: float = field(default_factory=time.time)
    lesson_complete: bool = False


class LessonSession:
    def __init__(self, user_id, lesson_df, lesson_id, native_lang, target_lang):
        self.df      = lesson_df.reset_index(drop=True)
        self.logger  = SessionLogger(user_id)
        self.state   = SessionState(
            user_id=user_id, lesson_id=lesson_id,
            native_lang=native_lang, target_lang=target_lang,
        )
        # Tracks when each step started — set explicitly from app.py
        self._step_start: dict[int, float] = {}

    # ── Step timing ──────────────────────────────────────────────────────

    def start_step(self, step: int):
        """Call this when a step begins to start its timer."""
        self._step_start[step] = time.time()

    def elapsed_ms(self, step: int) -> int:
        """Milliseconds since start_step(step) was called."""
        start = self._step_start.get(step, self.state.step_start_ts)
        return int((time.time() - start) * 1000)

    # ── Navigation ───────────────────────────────────────────────────────

    def next_phrase(self):
        self.state.phrase_index  += 1
        self.state.step_start_ts  = time.time()
        if self.state.phrase_index >= len(self.df):
            self.next_step()

    def next_step(self):
        self.state.current_step  += 1
        self.state.phrase_index   = 0
        self.state.step_start_ts  = time.time()
        if self.state.current_step > 7:
            self.state.lesson_complete = True

    def go_to_step(self, step: int):
        self.state.current_step  = step
        self.state.phrase_index  = 0
        self.state.step_start_ts = time.time()
        if step > 7:
            self.state.lesson_complete = True

    # ── Getters ──────────────────────────────────────────────────────────

    def phrases(self):
        return self.df.to_dict("records")

    def current_phrase(self):
        if self.state.phrase_index >= len(self.df):
            return self.df.iloc[-1].to_dict()
        return self.df.iloc[self.state.phrase_index].to_dict()

    def progress(self):
        return self.state.phrase_index + 1, len(self.df)

    def total(self):
        return len(self.df)

    # ── Evaluation ───────────────────────────────────────────────────────

    def score(self, user_input: str, expected: str,
              step: int | None = None,
              phrase_id: int | None = None) -> dict:
        """
        Evaluate user_input against expected and log the result.

        step      — pass explicitly from app.py (e.g. 1, 3, 6...)
        phrase_id — pass phrase["phrase_id"] explicitly, or None for whole-lesson steps
        """
        actual_step = step if step is not None else self.state.current_step
        elapsed     = self.elapsed_ms(actual_step)

        result = evaluate(user_input, expected)

        # For whole-lesson steps (1,4,6,7) phrase_id=0 means "whole lesson"
        log_phrase_id = phrase_id if phrase_id is not None else 0

        self.logger.log(
            lesson_id        = self.state.lesson_id,
            phrase_id        = log_phrase_id,
            step             = actual_step,
            similarity       = result["score"],
            response_time_ms = elapsed,
            attempts         = 1,
            success          = result["passed"],
            mode             = "lesson",
        )
        return result

    def score_phrase(self, user_input: str, phrase: dict, step: int) -> dict:
        """Convenience wrapper for single-phrase scoring."""
        return self.score(
            user_input = user_input,
            expected   = phrase["target"],
            step       = step,
            phrase_id  = int(phrase.get("phrase_id", 0)),
        )
