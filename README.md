# English Speaking Coach

A web app that helps Grade 7 students practise spoken English. A student records
or uploads a short speech, and the app transcribes it, checks grammar, measures
pacing, and scores it against the school's speaking rubric.

## How it works

1. **Whisper** transcribes the audio with word-level timestamps.
2. **Pacing analysis** measures words per minute, filler words (um, uh, like...)
   and long pauses, and flags segments that are too slow or too rushed.
3. **LanguageTool** finds grammar mistakes.
4. **Gemini** scores the speech on four rubric criteria and writes feedback:
   - Content and Coherence
   - Fluency
   - Accuracy and Pronunciation
   - Expression
5. The result is saved to a local SQLite database so a student's progress can be
   viewed on the dashboard.

"Confidence and Body Language" is part of the full school rubric but needs
video, so it is not scored. The four scored criteria are summed (max 8) and
rescaled to a total out of 10.

## Requirements

- Python 3.12
- [FFmpeg](https://ffmpeg.org/download.html) on your `PATH` (Whisper needs it to read audio)
- Java 8+ (LanguageTool runs on Java; it is downloaded on first start)
- A [Gemini API key](https://aistudio.google.com/apikey)
- Optional: an NVIDIA GPU with a CUDA build of PyTorch for much faster transcription

## Setup

```bash
python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Set your Gemini key, then start the server:

```powershell
# PowerShell
$env:GEMINI_API_KEY = "your-key-here"
uvicorn main:app --reload
```

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. The first start is slow because it downloads the
Whisper model and LanguageTool.

## Configuration

All settings are environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | none (required) | Gemini API key. Analysis fails without it. |
| `WHISPER_MODEL` | `small` | Whisper model size. Use `base.en` or `tiny.en` for faster, less accurate results. |
| `WHISPER_DEVICE` | auto | Force `cuda` or `cpu`. |
| `WHISPER_CONCURRENCY` | `1` | How many transcriptions may run at once. Keep at 1 on small GPUs. |
| `GEMINI_TIMEOUT_SEC` | `25` | Timeout for each Gemini call. |
| `GEMINI_MAX_RETRIES` | `5` | Retries before the request fails with a 502. |
| `GEMINI_RETRY_BASE_SEC` | `2` | Base delay between retries. |

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/analyze` | Form fields `student_name`, `roll_number`, `grade`, `section`, plus an `audio` file. Returns the analysis and saves the session. |
| `GET` | `/api/students` | Lists all students with their session counts. |
| `GET` | `/api/students/{roll_number}` | Returns one student's session history. |
| `GET` | `/` | Sign-in and recording page. |
| `GET` | `/dashboard.html` | Progress dashboard. |

## Project layout

```
main.py            FastAPI app, analysis pipeline, database models
static/            Front end (index.html, dashboard.html, app.js, styles.css)
audio_uploads/     Saved recordings (created automatically, not committed)
speaking_coach.db  SQLite database (created automatically, not committed)
```

## Notes

- If the database schema is out of date, the app moves the old file to
  `speaking_coach_old_schema_backup.db` and starts a new one.
- There is currently no login on the API, so run it only on a trusted network
  when it holds real student data.
