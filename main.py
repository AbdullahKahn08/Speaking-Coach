"""
English Speaking Coach - FastAPI Backend
-----------------------------------------
Serves the Modernist HTML frontend and exposes JSON APIs for analysis.

Run:
    pip install fastapi uvicorn python-multipart openai-whisper language-tool-python google-genai sqlalchemy
    set GEMINI_API_KEY=AIza...
    uvicorn main:app --reload

Then open: http://localhost:8000
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import whisper
import language_tool_python
from google import genai
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = BASE_DIR / "audio_uploads"
AUDIO_DIR.mkdir(exist_ok=True)

DB_PATH = "sqlite:///speaking_coach.db"

FILLER_WORDS = {"uh", "um", "hmm", "like", "you know", "err", "ah", "eh"}
PAUSE_SHORT, PAUSE_MEDIUM, PAUSE_LONG = 0.25, 0.75, 1.5

# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    grade = Column(String)
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
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ---------------------------------------------------------------------------
# MODELS (module-level, loaded once at startup)
# ---------------------------------------------------------------------------
print("Loading Whisper...")
_whisper = whisper.load_model("small")
print("Loading LanguageTool...")
_lt = language_tool_python.LanguageTool("en-US")
print("Loading Gemini client...")
_gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
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


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------
def analyze_audio(audio_path: str) -> dict:
    result = _whisper.transcribe(audio_path, word_timestamps=True, language="en")
    transcript = result["text"].strip()

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(),
                          "start": w["start"], "end": w["end"]})

    # Pauses
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

    lower = transcript.lower()
    fillers = []
    for f in FILLER_WORDS:
        fillers.extend([f] * len(re.findall(rf"\b{re.escape(f)}\b", lower)))

    # Grammar (LanguageTool)
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

    # Semantic review (Gemini)
    corrected = transcript
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
                grammar.append({
                    "rule_id": g.get("rule_id", "GENERAL"),
                    "wrong": g.get("wrong", ""),
                    "correction": g.get("correction", ""),
                    "message": g.get("message", ""),
                })
        except Exception as e:
            print(f"Gemini review skipped: {e}")

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
    }


def save_session(student_name: str, grade: str, audio_path: str, analysis: dict) -> int:
    db = SessionLocal()
    try:
        student = db.query(Student).filter_by(name=student_name).first()
        if not student:
            student = Student(name=student_name, grade=grade)
            db.add(student)
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


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="English Speaking Coach")


@app.post("/api/analyze")
async def analyze(
    student_name: str = Form(...),
    grade: str = Form(""),
    audio: UploadFile = File(...),
):
    if not student_name.strip():
        raise HTTPException(400, "student_name required")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"\W+", "_", student_name)
    save_path = AUDIO_DIR / f"{safe}_{ts}_{audio.filename}"
    with open(save_path, "wb") as f:
        f.write(await audio.read())

    analysis = analyze_audio(str(save_path))
    session_id = save_session(student_name, grade, str(save_path), analysis)
    analysis["session_id"] = session_id
    return analysis


@app.get("/api/students")
def list_students():
    db = SessionLocal()
    try:
        return [{"id": s.id, "name": s.name, "grade": s.grade,
                 "session_count": len(s.sessions)}
                for s in db.query(Student).all()]
    finally:
        db.close()


@app.get("/api/students/{name}")
def student_detail(name: str):
    db = SessionLocal()
    try:
        student = db.query(Student).filter_by(name=name).first()
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
            "grade": student.grade,
            "sessions": sessions_out,
            "top_rules": [{"rule": r, "count": c} for r, c in top_rules],
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# STATIC FRONTEND
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")