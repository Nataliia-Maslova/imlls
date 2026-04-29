# IMLLS — Intelligent Multilingual Language Learning System
### MVP for Master's Thesis in Data Science & Analytics

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For Whisper you also need `ffmpeg`:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 2. Place your database

Copy your Excel file into the `data/` folder and rename it:
```
data/imlls_database.xlsx
```

The file must have a sheet named `phrases` with columns:
`lesson_id, phrase_id, difficulty, en, uk, es, ko`

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
imlls/
├── app.py                  ← Main Streamlit UI
├── requirements.txt
├── data/
│   └── imlls_database.xlsx ← Your phrase database
├── engine/
│   ├── loader.py           ← Excel reader + language filtering
│   ├── scorer.py           ← Text normalization + RapidFuzz similarity
│   ├── tts.py              ← Google TTS with caching
│   ├── stt.py              ← OpenAI Whisper speech-to-text
│   ├── session.py          ← 8-step state machine
│   ├── logger.py           ← CSV interaction logger
│   └── adaptive.py         ← Cold start → ML adaptive engine
├── audio_cache/            ← Cached TTS audio files
└── logs/                   ← Per-user interaction CSV logs
```

---

## 8-Step Learning Loop

| Step | Name              | Shows native | Shows target | Voice |
|------|-------------------|:---:|:---:|:---:|
| 1    | Instruction       | ✓   | ✓   | –   |
| 2    | Listen & Repeat   | ✓   | ✓   | ✓   |
| 3    | Listen & Read     | –   | ✓   | ✓   |
| 4    | Multiple Choice   | –   | ✓   | –   |
| 5    | Speed Reading     | –   | ✓   | ✓   |
| 6    | Shadowing         | ✓   | –   | ✓   |
| 7    | Active Translation| ✓   | –   | ✓   |
| 8    | Speed Translation | ✓   | –   | ✓   |

---

## Adaptive ML Engine

- **Cold start** (< 25 interactions): always runs all 8 steps
- **Adaptive** (≥ 25 interactions): RandomForest predicts p(success on steps 7–8)
  - p ≥ 0.75 → skip steps 2–3 (easy path)
  - p < 0.45 → double repetition on step 2 (reinforcement path)
  - else → standard path

---

## Voice Input

The app uses **file upload** for voice input (works in any browser):
1. Record audio on your phone / use [vocaroo.com](https://vocaroo.com)
2. Upload the file
3. Whisper transcribes it locally

**Whisper model sizes** (edit `engine/stt.py` → `_model_size`):
- `tiny` — fastest (~1s), good enough for clear speech
- `base` — balanced (recommended for demo)
- `small` — more accurate, slower (~5-8s on CPU)

---

## Deploying to Streamlit Cloud (free)

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file to `app.py`
4. Add your `imlls_database.xlsx` to the repo

> Note: Whisper `tiny` model works on Streamlit Cloud free tier.
> Add `packages.txt` with `ffmpeg` for audio support.

---

## For the Thesis

Interaction logs are saved to `logs/{user_id}.csv` with columns:
```
timestamp, user_id, lesson_id, phrase_id, step,
similarity, response_time_ms, attempts, success, mode
```

Use these CSVs in a Jupyter notebook for:
- Similarity score by step (bar chart)
- Learning curve over time (line chart)
- Cold start vs Adaptive comparison (t-test)
- ML model feature importance

---

## packages.txt (for Streamlit Cloud)
```
ffmpeg
```
