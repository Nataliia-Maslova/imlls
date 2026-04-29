"""
Adaptive ML Engine.

Cold start  (< MIN_SAMPLES interactions): always runs full 8-step sequence.
Adaptive    (>= MIN_SAMPLES):             RandomForest predicts p(success on steps 7-8)
                                          and adjusts the step sequence accordingly.
"""
import numpy as np

MIN_SAMPLES = 25   # interactions needed before ML kicks in

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


# Full default 8-step sequence (1-indexed)
FULL_SEQUENCE   = [1, 2, 3, 4, 5, 6, 7, 8]
SKIP_EASY       = [1, 4, 5, 6, 7, 8]          # skip listen+repeat steps
EXTRA_REPEAT    = [1, 2, 2, 3, 4, 5, 6, 7, 8] # double repetition for weak phrases


class AdaptiveEngine:
    def __init__(self):
        self.mode       = "cold_start"
        self._model     = None
        self._scaler    = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_steps(self, phrase_features: dict | None = None) -> list[int]:
        """
        Return the list of step indices to run for the next phrase.
        phrase_features: dict with keys avg_similarity, avg_response_time,
                         total_attempts  (used only in adaptive mode)
        """
        if self.mode == "cold_start" or not self._is_trained:
            return FULL_SEQUENCE.copy()

        p = self._predict(phrase_features or {})

        if p >= 0.75:
            return SKIP_EASY.copy()
        elif p < 0.45:
            return EXTRA_REPEAT.copy()
        else:
            return FULL_SEQUENCE.copy()

    def maybe_train(self, log_rows: list[dict]) -> bool:
        """
        Try to train the model if enough data is available.
        Returns True if the model was (re)trained.
        """
        if not SK_AVAILABLE or len(log_rows) < MIN_SAMPLES:
            return False

        X, y = self._build_dataset(log_rows)
        if len(X) < 5 or len(set(y)) < 2:
            return False   # need at least 2 classes

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X_scaled, y)

        self._model      = clf
        self._scaler     = scaler
        self._is_trained = True
        self.mode        = "adaptive"
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_dataset(self, log_rows: list[dict]):
        """
        Build (X, y) from the session log.
        One sample per phrase_id.
        y = 1 if the user passed step 7 or 8 on first attempt.
        """
        from collections import defaultdict
        by_phrase = defaultdict(list)
        for row in log_rows:
            try:
                by_phrase[row["phrase_id"]].append(row)
            except KeyError:
                pass

        X, y = [], []
        for phrase_id, rows in by_phrase.items():
            sims   = [float(r["similarity"])       for r in rows]
            times  = [float(r["response_time_ms"]) for r in rows]
            steps  = [int(r["step"])               for r in rows]

            avg_sim   = np.mean(sims)   if sims  else 0.5
            avg_time  = np.mean(times)  if times else 3000
            n_attempts = len(rows)

            # label: did user succeed on step 7 or 8?
            late_rows  = [r for r in rows if int(r["step"]) >= 7]
            label = 1 if any(int(r["success"]) == 1 for r in late_rows) else 0

            X.append([avg_sim, avg_time, n_attempts])
            y.append(label)

        return np.array(X), np.array(y)

    def _predict(self, features: dict) -> float:
        """Return p(success) for a phrase given its aggregate features."""
        if not self._is_trained:
            return 0.5
        x = np.array([[
            features.get("avg_similarity",    0.5),
            features.get("avg_response_time", 3000),
            features.get("total_attempts",    4),
        ]])
        x_scaled = self._scaler.transform(x)
        return float(self._model.predict_proba(x_scaled)[0][1])
