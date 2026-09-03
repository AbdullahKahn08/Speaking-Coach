import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import whisper
import language_tool_python
from google import genai
from google.genai import types as genai_types
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = BASE_DIR / "audio_uploads"
AUDIO_DIR.mkdir(exist_ok=True)

DB_FILE = BASE_DIR / "speaking_coach.db"
DB_PATH = f"sqlite:///{DB_FILE}"

FILLER_WORDS = {"uh", "um", "hmm", "like", "you know", "err", "ah", "eh"}
PAUSE_SHORT, PAUSE_MEDIUM, PAUSE_LONG = 0.25, 0.75, 1.5

# Pacing thresholds (words/minute) used to flag segments as too slow or rushed.
# MIN_SEGMENT_SEC avoids flagging tiny fragments where wpm is noisy.
SLOW_WPM, FAST_WPM = 100, 190
MIN_SEGMENT_SEC = 1.2
CONTEXT_SPAN = 4  # words of context shown around a flagged pause

Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    roll_number = Column(String, unique=True, nullable=False)
    grade = Column(String)
    section = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("Session", back_populates="student")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    audio_path = Column(String)
    transcript = Column(Text)
    duration_sec = Column(Float)
    wpm = Column(Float)
    filler_count = Column(Integer)
    short_pauses = Column(Integer)
    medium_pauses = Column(Integer)
    long_pauses = Column(Integer)
    avg_pause_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    student = relationship("Student", back_populates="sessions")
    mistakes = relationship("Mistake", back_populates="session")

class Mistake(Base):
    __tablename__ = "mistakes"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    category = Column(String)
    rule_id = Column(String)
    wrong_text = Column(String)
    correction = Column(String)
    explanation = Column(Text)
    session = relationship("Session", back_populates="mistakes")

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})

# The database file predates the roll_number/section columns added to the
# Student model. create_all() only creates *missing* tables, it never alters
# existing ones, so an old file left every request failing with "no such
# column". Detect that mismatch here and move the old file aside instead of
# crashing — a fresh DB is created automatically, and nothing is lost since
# the old file is kept as a backup.
inspector = inspect(engine)
if "students" in inspector.get_table_names():
    existing_cols = {c["name"] for c in inspector.get_columns("students")}
    required_cols = {c.name for c in Student.__table__.columns}
    if not required_cols.issubset(existing_cols):
        engine.dispose()
        backup_path = DB_FILE.with_name(DB_FILE.stem + "_old_schema_backup.db")
        if backup_path.exists():
            backup_path.unlink()
        DB_FILE.rename(backup_path)
        print(f"[startup] Database schema was out of date (missing "
              f"{required_cols - existing_cols}). Old data backed up to "
              f"'{backup_path.name}'; starting a fresh database.")
        engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# "small" is accurate but slow on CPU. Override with WHISPER_MODEL=base.en or
# tiny.en for a large speed boost (English-only models also skip language
# detection). See the "why is this slow" notes near analyze_audio().
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

print(f"Loading Whisper ({WHISPER_MODEL})...")
_whisper = whisper.load_model(WHISPER_MODEL)
print(f"Whisper device: {_whisper.device}")  # 'cuda' if a GPU was found, else 'cpu'
print("Loading LanguageTool...")
_lt = language_tool_python.LanguageTool("en-US")
print("Loading Gemini client...")
# http_options timeout is a first line of defense; the hard backstop is the
# future.result(timeout=...) guard around the call in _run_gemini_review,
# since some SDK versions don't always honor http_options reliably.
GEMINI_TIMEOUT_SEC = float(os.environ.get("GEMINI_TIMEOUT_SEC", "25"))
_gemini = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    http_options=genai_types.HttpOptions(timeout=int(GEMINI_TIMEOUT_SEC * 1000)),
)
print("Ready.")


LLM_REVIEW_PROMPT = """You are a kind English teacher reviewing what a school student said (transcribed from audio).

The student's transcript:
\"\"\"{transcript}\"\"\"

Find EVERY issue - especially the ones a rule-based checker would miss:
  - wrong verb tense
  - wrong word choice / semantic errors
  - awkward sentence structure, run-ons, or fragments
  - missing/wrong articles or prepositions
  - subject-verb agreement

Return ONLY a JSON object (no prose, no markdown fences) with this exact shape:
{{
  "corrected": "the full transcript rewritten correctly, preserving the student's meaning",
  "mistakes": [
    {{
      "rule_id": "short uppercase category, one of: VERB_TENSE, WORD_CHOICE, ARTICLE, PREPOSITION, SUBJECT_VERB_AGREEMENT, WORD_ORDER, RUN_ON, FRAGMENT, PLURAL",
      "wrong": "the exact wrong phrase from the transcript",
      "correction": "the corrected phrase",
      "message": "one short sentence explaining the mistake in kid-friendly language"
    }}
  ]
}}

If truly no issues, return {{"corrected": "<original transcript unchanged>", "mistakes": []}}."""


def _fmt_time(seconds: float) -> str:
    m, s = divmod(max(0, int(round(seconds))), 60)
    return f"{m}:{s:02d}"


def _word_context(words: List[dict], index: int, span: int = CONTEXT_SPAN):
    """Text just before/after words[index], for locating a pause in the transcript."""
    before = " ".join(w["word"] for w in words[max(0, index - span):index]).strip()
    after = " ".join(w["word"] for w in words[index:index + span]).strip()
    return before, after


def build_pacing_notes(words: List[dict], segments: List[dict]) -> List[dict]:
    """Locate *where* in the recording the speaker paused too long or slowed
    down/rushed, so feedback can point at specific moments instead of just
    aggregate counts."""
    notes = []

    # Long pauses, anchored to the words on either side of the gap.
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= PAUSE_LONG:
            before, after = _word_context(words, i)
            notes.append({
                "type": "long_pause",
                "time_sec": words[i - 1]["end"],
                "time": _fmt_time(words[i - 1]["end"]),
                "duration_sec": round(gap, 2),
                "before": before,
                "after": after,
                "message": f"Paused for {gap:.1f}s" + (f' after "{before}"' if before else ""),
            })

    # Slow / rushed stretches, using Whisper's own segment boundaries (which
    # already tend to break on pauses/sentences, so local wpm is meaningful).
    for seg in segments:
        text = seg.get("text", "").strip()
        seg_words = text.split()
        dur = seg.get("end", 0) - seg.get("start", 0)
        if not seg_words or dur < MIN_SEGMENT_SEC:
            continue
        seg_wpm = len(seg_words) / dur * 60
        if seg_wpm < SLOW_WPM:
            notes.append({
                "type": "slow",
                "time_sec": seg["start"],
                "time": _fmt_time(seg["start"]),
                "wpm": round(seg_wpm),
                "text": text,
                "message": f"Spoke slowly here (~{round(seg_wpm)} wpm)",
            })
        elif seg_wpm > FAST_WPM:
            notes.append({
                "type": "fast",
                "time_sec": seg["start"],
                "time": _fmt_time(seg["start"]),
                "wpm": round(seg_wpm),
                "text": text,
                "message": f"Rushed through this part (~{round(seg_wpm)} wpm)",
            })

    notes.sort(key=lambda n: n["time_sec"])
    return notes


def _run_languagetool(transcript: str) -> list:
    grammar = []
    for m in _lt.check(transcript):
        if m.rule_id.startswith("MORFOLOGIK"):
            continue
        grammar.append({
            "rule_id": m.rule_id,
            "wrong": transcript[m.offset:m.offset + m.error_length],
            "correction": m.replacements[0] if m.replacements else "",
            "message": m.message,
        })
    return grammar


def _run_gemini_review(transcript: str):
    """Returns (corrected_transcript, extra_mistakes)."""
    corrected = transcript
    mistakes = []
    if transcript and os.environ.get("GEMINI_API_KEY"):
        try:
            resp = _gemini.models.generate_content(
                model="gemini-3.6-flash",
                contents=LLM_REVIEW_PROMPT.format(transcript=transcript),
            )
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip()).strip()
            parsed = json.loads(raw)
            corrected = parsed.get("corrected", transcript)
            for g in parsed.get("mistakes", []):
                mistakes.append({
                    "rule_id": g.get("rule_id", "GENERAL"),
                    "wrong": g.get("wrong", ""),
                    "correction": g.get("correction", ""),
                    "message": g.get("message", ""),
                })
        except Exception as e:
            print(f"Gemini review skipped: {e}")
    return corrected, mistakes


# LanguageTool and Gemini both only need the transcript, so run them
# concurrently instead of back-to-back — this is usually the single biggest
# easy win since Gemini is a network call that can take several seconds.
_review_executor = ThreadPoolExecutor(max_workers=4)


def analyze_audio(audio_path: str) -> dict:
    t_start = time.time()

    result = _whisper.transcribe(audio_path, word_timestamps=True, language="en")
    transcript = result["text"].strip()
    segments = result.get("segments", [])
    t_whisper = time.time()
    print(f"[timing] whisper transcription: {t_whisper - t_start:.1f}s")

    words = []
    for seg in segments:
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(),
                          "start": w["start"], "end": w["end"]})

    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i-1]["end"]
        if gap >= PAUSE_SHORT:
            pauses.append(gap)
    short = sum(1 for p in pauses if PAUSE_SHORT <= p < PAUSE_MEDIUM)
    medium = sum(1 for p in pauses if PAUSE_MEDIUM <= p < PAUSE_LONG)
    long_ = sum(1 for p in pauses if p >= PAUSE_LONG)
    avg_pause_ms = (sum(pauses) / len(pauses) * 1000) if pauses else 0

    duration = words[-1]["end"] if words else 0
    wpm = (len(words) / duration * 60) if duration > 0 else 0
    pacing_notes = build_pacing_notes(words, segments)

    lower = transcript.lower()
    fillers = []
    for f in FILLER_WORDS:
        fillers.extend([f] * len(re.findall(rf"\b{re.escape(f)}\b", lower)))

    lt_future = _review_executor.submit(_run_languagetool, transcript)
    gemini_future = _review_executor.submit(_run_gemini_review, transcript)
    grammar = lt_future.result()
    try:
        corrected, gemini_mistakes = gemini_future.result(timeout=GEMINI_TIMEOUT_SEC)
        grammar.extend(gemini_mistakes)
    except FutureTimeoutError:
        print(f"[timing] Gemini review exceeded {GEMINI_TIMEOUT_SEC}s — "
              f"skipping it for this session (LanguageTool results still used).")
        corrected = transcript
    t_review = time.time()
    print(f"[timing] grammar review (LanguageTool + Gemini, parallel): {t_review - t_whisper:.1f}s")
    print(f"[timing] total: {t_review - t_start:.1f}s")

    return {
        "transcript": transcript,
        "corrected": corrected,
        "duration": duration,
        "wpm": wpm,
        "word_count": len(words),
        "short_pauses": short,
        "medium_pauses": medium,
        "long_pauses": long_,
        "avg_pause_ms": avg_pause_ms,
        "fillers": fillers,
        "grammar_mistakes": grammar,
        "pacing_notes": pacing_notes,
    }


def save_session(student_name: str, roll_number: str, grade: str, section: str,
                  audio_path: str, analysis: dict) -> int:
    db = SessionLocal()
    try:
        student = db.query(Student).filter_by(roll_number=roll_number).first()
        if not student:
            student = Student(name=student_name, roll_number=roll_number,
                              grade=grade, section=section)
            db.add(student)
            db.commit()
        else:
            # Keep the record current if the student re-enters with updated details.
            student.name, student.grade, student.section = student_name, grade, section
            db.commit()

        session = Session(
            student_id=student.id,
            audio_path=audio_path,
            transcript=analysis["transcript"],
            duration_sec=analysis["duration"],
            wpm=analysis["wpm"],
            filler_count=len(analysis["fillers"]),
            short_pauses=analysis["short_pauses"],
            medium_pauses=analysis["medium_pauses"],
            long_pauses=analysis["long_pauses"],
            avg_pause_ms=analysis["avg_pause_ms"],
        )
        db.add(session)
        db.commit()

        for g in analysis["grammar_mistakes"]:
            db.add(Mistake(session_id=session.id, category="grammar",
                           rule_id=g["rule_id"], wrong_text=g["wrong"],
                           correction=g["correction"], explanation=g["message"]))
        for f in analysis["fillers"]:
            db.add(Mistake(session_id=session.id, category="filler",
                           rule_id="FILLER", wrong_text=f, correction="",
                           explanation=f"Filler word used: '{f}'"))
        db.commit()
        return session.id
    finally:
        db.close()


app = FastAPI(title="English Speaking Coach")


@app.post("/api/analyze")
async def analyze(
    student_name: str = Form(...),
    roll_number: str = Form(...),
    grade: str = Form(""),
    section: str = Form(""),
    audio: UploadFile = File(...),
):
    if not student_name.strip():
        raise HTTPException(400, "student_name required")
    if not roll_number.strip():
        raise HTTPException(400, "roll_number required")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"\W+", "_", f"{roll_number}_{student_name}")
    save_path = AUDIO_DIR / f"{safe}_{ts}_{audio.filename}"
    with open(save_path, "wb") as f:
        f.write(await audio.read())

    analysis = analyze_audio(str(save_path))
    session_id = save_session(student_name, roll_number, grade, section,
                               str(save_path), analysis)
    analysis["session_id"] = session_id
    return analysis


@app.get("/api/students")
def list_students():
    db = SessionLocal()
    try:
        return [{"id": s.id, "name": s.name, "roll_number": s.roll_number,
                 "grade": s.grade, "section": s.section,
                 "session_count": len(s.sessions)}
                for s in db.query(Student).all()]
    finally:
        db.close()


@app.get("/api/students/{roll_number}")
def student_detail(roll_number: str):
    db = SessionLocal()
    try:
        student = db.query(Student).filter_by(roll_number=roll_number).first()
        if not student:
            raise HTTPException(404, "not found")

        sessions_out = []
        rule_counts = {}
        for s in student.sessions:
            grammar_ms = [m for m in s.mistakes if m.category == "grammar"]
            for m in grammar_ms:
                rule_counts[m.rule_id] = rule_counts.get(m.rule_id, 0) + 1
            sessions_out.append({
                "id": s.id,
                "date": s.created_at.strftime("%Y-%m-%d %H:%M"),
                "wpm": round(s.wpm, 1),
                "duration": round(s.duration_sec, 1),
                "fillers": s.filler_count,
                "long_pauses": s.long_pauses,
                "grammar_mistakes": len(grammar_ms),
                "transcript": s.transcript,
            })
        top_rules = sorted(rule_counts.items(), key=lambda x: -x[1])[:10]
        return {
            "name": student.name,
            "roll_number": student.roll_number,
            "grade": student.grade,
            "section": student.section,
            "sessions": sessions_out,
            "top_rules": [{"rule": r, "count": c} for r, c in top_rules],
        }
    finally:
        db.close()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
@app.get("/index.html")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard.html")
def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")