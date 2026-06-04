# IMLLS — Intelligent Multilingual Language Learning System
### Master's Thesis Project · Data Science & Analytics

---

## Overview

IMLLS is a Streamlit-based language learning app that guides learners through an **8-step structured practice loop** with adaptive difficulty, speech recognition, grammatical error correction, and gamification. Supports English, Ukrainian, Spanish, and Korean.

**Live demo:** [Streamlit Cloud](#) *(replace with your URL)*

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For Whisper voice input, install `ffmpeg`:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### 2. Configure secrets

Copy `.streamlit/secrets_template.toml` → `.streamlit/secrets.toml` and fill in your credentials (Google Sheets API key, etc.).

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Learning Modules

| Module | Description | Data source |
|--------|-------------|-------------|
| **Grammar** | 8-step phrase practice with GEC correction | `data/imlls_database.xlsx` |
| **Vocabulary** | Topic-based word learning (Family, Food, Travel…) | `data/vocabulary.xlsx` |
| **Reading** | IPA-guided reading with audio playback | `data/reading_lessons.xlsx` |
| **My Phrases** | Custom lessons — same 8-step flow | `data/custom_phrases.csv` |

---

## 8-Step Learning Loop

| Step | Name | Native shown | Target shown | Voice |
|------|------|:---:|:---:|:---:|
| 1 | Introduction | ✓ | ✓ | – |
| 2 | Listen & Repeat | ✓ | ✓ | ✓ |
| 3 | Listen & Read | – | ✓ | ✓ |
| 4 | Multiple Choice | – | ✓ | – |
| 5 | Speed Reading | – | ✓ | ✓ |
| 6 | Shadowing | ✓ | – | ✓ |
| 7 | Active Translation | ✓ | – | ✓ |
| 8 | Speed Translation + GEC | ✓ | – | ✓ |

---

## Project Structure

```
imlls/
├── app.py                    ← Main launcher (module selector + progress cards)
├── grammar.py                ← Grammar & Vocabulary practice (8-step loop)
├── reading_app.py            ← Reading module
├── custom_app.py             ← My Phrases module
├── requirements.txt
├── packages.txt              ← System packages for Streamlit Cloud (ffmpeg)
│
├── engine/
│   ├── loader.py             ← Excel reader + language filtering
│   ├── vocab_loader.py       ← Vocabulary Excel reader
│   ├── scorer.py             ← RapidFuzz text similarity scoring
│   ├── analyzer.py           ← Semantic scoring via sentence-transformers (MiniLM)
│   ├── gec.py                ← Grammatical Error Correction (T5, HuggingFace)
│   ├── tts.py                ← Google TTS with local file caching
│   ├── stt.py                ← OpenAI Whisper speech-to-text
│   ├── session.py            ← 8-step state machine
│   ├── adaptive.py           ← Adaptive engine (RandomForest, cold-start logic)
│   ├── logger.py             ← CSV interaction logger + progress tracker
│   ├── gamification.py       ← Streak, XP, levels, badges
│   ├── picker.py             ← Lesson navigation UI components
│   ├── character_widget.py   ← Animated character sidebar widget
│   ├── characters.py         ← Character data and phrase helpers
│   ├── auth_store.py         ← User authentication (YAML-based)
│   ├── custom_store.py       ← Custom lesson CRUD
│   └── i18n.py               ← Localized UI strings (en/uk/es/ko)
│
├── data/
│   ├── imlls_database.xlsx   ← Grammar phrase database
│   ├── vocabulary.xlsx       ← Vocabulary by topic
│   ├── reading_lessons.xlsx  ← Reading lessons with IPA
│   ├── custom_phrases.csv    ← User-created custom lessons
│   └── gamification.csv      ← XP / streak / badge records
│
├── static/
│   ├── lesson_images/        ← Lesson illustrations (JPEG, ~23 MB)
│   ├── app_images/           ← Module banner images (JPEG, ~4 MB)
│   └── mova/                 ← Mova design system CSS
│
├── audio_cache/              ← Cached TTS MP3 files
└── logs/                     ← Per-user interaction CSV logs
```

---

## AI & ML Components

### Adaptive Engine (`engine/adaptive.py`)
- **Cold start** (< 25 interactions): runs all 8 steps
- **Adaptive** (≥ 25 interactions): RandomForest predicts success probability on steps 7–8
  - p ≥ 0.75 → skip steps 2–3 (fast track)
  - p < 0.45 → double repetition on step 2 (reinforcement)
  - else → standard path

### Semantic Scoring (`engine/analyzer.py`)
- `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) for multilingual cosine similarity
- POS structure analysis (English, via NLTK)
- Relative score calibration against lesson phrase range
- Vocabulary coverage penalty

### Grammatical Error Correction (`engine/gec.py`)
- Fine-tuned T5 models hosted on HuggingFace Hub:
  - `natashasms/en-gec-model` (English)
  - `natashasms/es-gec-model` (Spanish)
  - `natashasms/ko-gec-model` (Korean)
- Lazy-loaded on first use; cached for the session

### Speech-to-Text (`engine/stt.py`)
- OpenAI Whisper (`base` model by default)
- Accepts uploaded audio files (WAV, WebM)
- Change `_model_size` in `stt.py` for speed/accuracy trade-off:
  - `tiny` — fastest (~1s), good for clear speech
  - `base` — balanced (default)
  - `small` — more accurate, slower on CPU

---

## Gamification

- **XP** earned per step and lesson completion
- **Streaks** tracked daily
- **Badges** unlocked at milestones
- **Levels** with named tiers based on total XP
- Progress visible in the sidebar widget

---

## Deploying to Streamlit Cloud (free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect repo → set main file to `app.py`
3. Add secrets via the Streamlit Cloud dashboard (copy from `secrets_template.toml`)

`packages.txt` already includes `ffmpeg` for audio support.

> **Note:** Whisper `base` model works on Streamlit Cloud free tier via `@st.cache_resource` (loaded once per deployment, not per user session).

---

## Logging & Thesis Data

Interaction logs → `logs/{user_id}.csv`:
```
timestamp, user_id, lesson_id, phrase_id, step,
similarity, response_time_ms, attempts, success, mode
```

Use with `analysis.ipynb` for:
- Similarity score distribution by step
- Learning curve over sessions
- Cold start vs. adaptive performance comparison (t-test)
- RandomForest feature importance
