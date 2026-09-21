import os
import re
import json
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
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
    score = Column(Float)         # rubric total /10, in 0.5 steps (see RUBRIC_PROMPT)
    score_reason = Column(Text)
    rubric_criteria = Column(Text)  # JSON: {criterion_key: {"score":.., "band":.., "comment":..}}
    strength = Column(Text)       # required "AI feedback" per the assessment spec
    improve = Column(Text)
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

from sqlalchemy import event
from sqlalchemy.engine import Engine

# SQLite's default journal mode only allows ONE writer at a time for the
# entire database file — if two students hit "Analyze" within the same
# instant, the second one gets a "database is locked" error. WAL (Write-
# Ahead Logging) mode lets reads and writes happen concurrently instead,
# covering the "a few people submit around the same time" case that matters
# for a classroom demo. It does NOT make SQLite handle true high-concurrency
# load (see deployment notes for when to move to Postgres) — it just removes
# the most common lock error at small scale. Applies to every connection
# SQLAlchemy opens, on both engines created below.
@event.listens_for(Engine, "connect")
def _enable_sqlite_wal(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")  # wait up to 5s instead of failing instantly
    cursor.close()

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})

# The database file predates the roll_number/section columns added to the
# Student model (and now the score/score_reason columns on Session).
# create_all() only creates *missing* tables, it never alters existing ones,
# so an old file left every request failing with "no such column". Detect
# that mismatch here and move the old file aside instead of crashing — a
# fresh DB is created automatically, and nothing is lost since the old file
# is kept as a backup.
inspector = inspect(engine)
_schema_stale = False
_missing_cols = set()
for _model in (Student, Session):
    table_name = _model.__tablename__
    if table_name in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
        required_cols = {c.name for c in _model.__table__.columns}
        if not required_cols.issubset(existing_cols):
            _schema_stale = True
            _missing_cols |= (required_cols - existing_cols)

if _schema_stale:
    engine.dispose()
    backup_path = DB_FILE.with_name(DB_FILE.stem + "_old_schema_backup.db")
    if backup_path.exists():
        backup_path.unlink()
    DB_FILE.rename(backup_path)
    print(f"[startup] Database schema was out of date (missing "
          f"{_missing_cols}). Old data backed up to "
          f"'{backup_path.name}'; starting a fresh database.")
    engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})

Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# "small" is accurate but slow on CPU. Override with WHISPER_MODEL=base.en or
# tiny.en for a large speed boost (English-only models also skip language
# detection). See the "why is this slow" notes near analyze_audio().
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# GPU acceleration: Whisper runs dramatically faster on an NVIDIA GPU via
# CUDA (often 5-10x). Auto-detects if a CUDA-capable GPU + the right PyTorch
# build are available; falls back to CPU otherwise. Force a specific device
# with WHISPER_DEVICE=cuda or WHISPER_DEVICE=cpu if the auto-detect guesses
# wrong (e.g. multiple GPUs, or you want to reserve the GPU for something else).
import torch

_forced_device = os.environ.get("WHISPER_DEVICE", "").strip().lower()
if _forced_device in ("cuda", "cpu"):
    WHISPER_DEVICE = _forced_device
elif torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
else:
    WHISPER_DEVICE = "cpu"

print(f"Loading Whisper ({WHISPER_MODEL}) on {WHISPER_DEVICE}...")
if WHISPER_DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
elif _forced_device != "cpu":
    print("  No CUDA GPU detected (or PyTorch was installed CPU-only) — "
          "running on CPU. See the GPU setup notes below load_model() if "
          "you have an NVIDIA GPU and want to use it.")

_whisper = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

# Only one Whisper model instance exists, and on GPU it lives in a small
# (4GB-class) pool of VRAM shared by everyone. If two requests call
# .transcribe() at literally the same instant from different threads, they
# can corrupt each other's GPU memory or throw CUDA out-of-memory errors.
# This semaphore makes concurrent transcriptions queue up and run one at a
# time — safe on any GPU size, and on CPU it just prevents needless
# thread-thrashing. Everything else (LanguageTool, Gemini, saving to the DB)
# still runs freely in parallel across users; only the actual GPU-bound
# Whisper call is serialized. Override via WHISPER_CONCURRENCY if you move
# to a bigger GPU that can genuinely run more than one transcription at once.
WHISPER_CONCURRENCY = int(os.environ.get("WHISPER_CONCURRENCY", "1"))
_whisper_semaphore = threading.Semaphore(WHISPER_CONCURRENCY)
print(f"Whisper device: {_whisper.device}")  # 'cuda' if a GPU was found, else 'cpu'
print("Loading LanguageTool...")
_lt = language_tool_python.LanguageTool("en-US")
print("Loading Gemini client...")
# http_options timeout is a first line of defense; the hard backstop is the
# future.result(timeout=...) guard around the call in _run_gemini_review,
# since some SDK versions don't always honor http_options reliably.
GEMINI_TIMEOUT_SEC = float(os.environ.get("GEMINI_TIMEOUT_SEC", "25"))
# Gemini review is required (see _run_gemini_review) — these control how hard
# it retries before finally giving up and failing the request.
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))
GEMINI_RETRY_BASE_SEC = float(os.environ.get("GEMINI_RETRY_BASE_SEC", "2"))
_gemini = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    http_options=genai_types.HttpOptions(timeout=int(GEMINI_TIMEOUT_SEC * 1000)),
)
print("Ready.")


# Single source of truth for the rubric text — used to BOTH build the Gemini
# prompt below AND power the /api/rubric endpoint that the dashboard reads to
# show "how this score is calculated". Keeping one copy means the table a
# teacher sees always matches exactly what Gemini was actually told to judge
# against — no risk of the two drifting apart after an edit.
RUBRIC_DEFINITIONS = {
    "content_coherence": {
        "label": "Content Relevancy & Coherence",
        "bands": {
            2.0: "Maintains relevance throughout without deviating; gives diverse ideas; develops the topic coherently with a proper start, middle and end.",
            1.5: "Content is on-topic but ideas are a bit limited, with adequate coherence.",
            1.0: "Misses some points and doesn't speak for the given time; some breakdowns in coherence.",
            0.5: "Insufficient or totally deviated content; breakdowns in coherence.",
        },
    },
    "fluency": {
        "label": "Fluency",
        "bands": {
            2.0: 'Speaks fluently with natural pauses and conversational fillers ("you know", "I mean", "well", "basically"), without repetition or self-correction.',
            1.5: 'Speaks fluently with little pauses ("um", "uh", "mm-hmm"), showing little hesitation, occasional repetition and self-correction.',
            1.0: "Cannot respond without noticeable pauses; may speak slowly, with frequent repetition and self-correction.",
            0.5: "Speaks with long pauses, repetitions and hesitation.",
        },
    },
    "accuracy_pronunciation": {
        "label": "Accuracy & Pronunciation",
        "bands": {
            2.0: "Produces consistently accurate grammatical structures apart from occasional slips; uses a full range of pronunciation features with precision; effortless to understand.",
            1.5: "Produces a majority of error-free sentences with only occasional basic errors; easy to understand with minimal pronunciation errors.",
            1.0: "May make frequent grammar mistakes though these rarely cause comprehension problems; can generally be understood, though mispronunciation of individual words reduces clarity at times.",
            0.5: "Errors are frequent and may lead to misunderstanding; mispronunciations are frequent and cause difficulty for the listener.",
        },
    },
    "expression": {
        "label": "Expression (Vocabulary & Cohesion)",
        "bands": {
            2.0: "Uses vocabulary with full flexibility and precision; doesn't repeat the same words/structures; uses a variety of words and structures; uses cohesive devices to connect sentences.",
            1.5: "Has a wide enough vocabulary to discuss the topic at length and make meaning clear, despite occasional inappropriate word choices; little use of cohesive devices.",
            1.0: "Uses vocabulary with limited flexibility and hardly any cohesive devices.",
            0.5: "Makes frequent errors in word choice; gives only simple responses; frequently unable to convey the basic message.",
        },
    },
}


def _build_rubric_prompt_section() -> str:
    """Renders RUBRIC_DEFINITIONS into the numbered block RUBRIC_PROMPT
    embeds, so the prompt text is generated from the same dict the
    /api/rubric endpoint serves — never maintained twice."""
    lines = []
    for i, (key, crit) in enumerate(RUBRIC_DEFINITIONS.items(), start=1):
        lines.append(f"{i}. {crit['label'].upper()}")
        for band in (2.0, 1.5, 1.0, 0.5):
            lines.append(f"  {band} - {crit['bands'][band]}")
        lines.append("")
    return "\n".join(lines).rstrip()


RUBRIC_PROMPT = """You are an English language assessor conducting a formal, rubric-based evaluation of a Grade 7 student's spoken English (transcribed from audio).

The student's transcript:
\"\"\"{transcript}\"\"\"

The student's fluency metrics for context:
  - Speaking rate: {wpm} words per minute (140-160 is typically fluent)
  - Filler words used: {filler_count}
  - Long pauses (over 1.5s): {long_pauses}

FAIRNESS RULE (mandatory): Do not penalize the speaker for having a Pakistani
or regional English accent. Judge pronunciation and intelligibility only -
whether the words are clear and understandable - never accent itself.

Score the response against these FOUR rubric criteria. For each one, select
the band (2.0, 1.5, 1.0, or 0.5) whose description best matches the response,
using the exact band definitions below - do not invent your own criteria or
scale.

""" + _build_rubric_prompt_section() + """

("Confidence and Body Language" is part of the full rubric but requires video
- eye contact, gestures, posture - which is not available from audio. Do not
attempt to score it; it is excluded from this evaluation entirely.)

Separately, identify EVERY grammar/structure issue for the mistake list below
- especially ones the rubric bands above wouldn't individually call out:
  - incorrect verb tense
  - incorrect word choice / semantic errors
  - awkward sentence structure, run-ons, or fragments
  - missing or incorrect articles or prepositions
  - subject-verb agreement

Write score_reason, each criterion's comment, and each mistake's message in a
formal, evaluative register, as an examiner would write on an assessment
report - not as an encouraging teacher praising a student. Concretely:
  - State observations plainly (e.g. "Speaking rate was within the fluent
    range; two grammatical errors were noted.") rather than praising effort.
  - Do not use exclamation marks.
  - Do not use words like "great", "fantastic", "excellent job", "well done",
    "nice work", or similar praise language.
  - Refer to "the response" or "the speaker", not "you".

Then write exactly one strength and one improvement point, each one short
sentence in simple English suitable for a Grade 7 student to read themselves
(this part only may be encouraging in tone, per the school's feedback format
- e.g. "Strength: The response answered the question clearly and gave good
reasons." / "Improve: Try to reduce long pauses and use a wider range of
vocabulary.").

Return ONLY a JSON object (no prose, no markdown fences) with this exact shape:
{{
  "criteria": {{
    "content_coherence": {{"band": 2.0, "comment": "one formal sentence justifying this band"}},
    "fluency": {{"band": 2.0, "comment": "one formal sentence justifying this band"}},
    "accuracy_pronunciation": {{"band": 2.0, "comment": "one formal sentence justifying this band"}},
    "expression": {{"band": 2.0, "comment": "one formal sentence justifying this band"}}
  }},
  "score_reason": "one concise, formal, evaluative sentence summarizing the overall basis for the score",
  "strength": "one short encouraging sentence in simple Grade 7 English",
  "improve": "one short constructive sentence in simple Grade 7 English",
  "corrected": "the full transcript rewritten correctly, preserving the student's meaning",
  "mistakes": [
    {{
      "rule_id": "short uppercase category, one of: VERB_TENSE, WORD_CHOICE, ARTICLE, PREPOSITION, SUBJECT_VERB_AGREEMENT, WORD_ORDER, RUN_ON, FRAGMENT, PLURAL",
      "wrong": "the exact wrong phrase from the transcript",
      "correction": "the corrected phrase",
      "message": "one concise, formal sentence explaining the mistake and the applicable rule"
    }}
  ]
}}

Each "band" value must be exactly one of 2.0, 1.5, 1.0, or 0.5 - no other
numbers. If truly no grammar issues beyond what the bands capture, still
score all four criteria and return "mistakes": []."""


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


CRITERIA_KEYS = ("content_coherence", "fluency", "accuracy_pronunciation", "expression")
VALID_BANDS = {2.0, 1.5, 1.0, 0.5}


def _run_gemini_review(transcript: str, wpm: float, filler_count: int, long_pauses: int):
    """Returns (corrected_transcript, extra_mistakes, score, score_reason,
    criteria, strength, improve).

    `criteria` is a dict of the 4 rubric bands actually scored (see
    RUBRIC_PROMPT) — "Confidence and Body Language" is excluded since it
    needs video. `score` is those 4 bands summed (max 8) and rescaled ×1.25
    to a /10 total, per the school's rubric, so an AI-only session score
    still means the same thing as the full 5-criteria in-person rubric.

    Gemini review is treated as REQUIRED, not optional: LanguageTool alone
    misses meaning-level mistakes, which is the whole reason Gemini was
    added. So instead of silently swallowing a single failure and shipping
    a session with half its review missing, this retries with exponential
    backoff, and only gives up (raising, so the request fails loudly) after
    GEMINI_MAX_RETRIES attempts.
    """
    if not transcript:
        return transcript, [], None, None, {}, "", ""
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Gemini review is required for this "
            "app to work — set the env var and restart the server."
        )

    prompt = RUBRIC_PROMPT.format(
        transcript=transcript,
        wpm=round(wpm),
        filler_count=filler_count,
        long_pauses=long_pauses,
    )

    last_err = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            resp = _gemini.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip()).strip()
            parsed = json.loads(raw)

            corrected = parsed.get("corrected", transcript)
            mistakes = [{
                "rule_id": g.get("rule_id", "GENERAL"),
                "wrong": g.get("wrong", ""),
                "correction": g.get("correction", ""),
                "message": g.get("message", ""),
            } for g in parsed.get("mistakes", [])]

            raw_criteria = parsed.get("criteria", {})
            criteria = {}
            raw_sum = 0.0
            for key in CRITERIA_KEYS:
                entry = raw_criteria.get(key, {})
                band = entry.get("band")
                # Snap defensively to the nearest valid band in case the
                # model drifts off the four allowed values.
                if not isinstance(band, (int, float)) or band not in VALID_BANDS:
                    band = min(VALID_BANDS, key=lambda v: abs(v - (band or 1.0))) \
                        if isinstance(band, (int, float)) else 1.0
                criteria[key] = {"band": band, "comment": entry.get("comment", "")}
                raw_sum += band

            # 4 criteria × 2.0 max = 8 raw points -> rescale to a /10 total,
            # rounded to the nearest 0.5 so it still reads as a rubric score.
            score = round((raw_sum / 8.0 * 10) * 2) / 2

            score_reason = parsed.get("score_reason", "")
            strength = parsed.get("strength", "")
            improve = parsed.get("improve", "")
            return corrected, mistakes, score, score_reason, criteria, strength, improve
        except Exception as e:
            last_err = e
            print(f"[gemini] attempt {attempt}/{GEMINI_MAX_RETRIES} failed: {e}")
            if attempt < GEMINI_MAX_RETRIES:
                wait = GEMINI_RETRY_BASE_SEC * (2 ** (attempt - 1))
                print(f"[gemini] retrying in {wait:.0f}s...")
                time.sleep(wait)

    # Every attempt failed — raise rather than quietly returning the
    # uncorrected transcript, so the caller (and the person testing the app)
    # sees a clear error instead of a session that's silently incomplete.
    raise RuntimeError(
        f"Gemini review failed after {GEMINI_MAX_RETRIES} attempts. "
        f"Last error: {last_err}"
    )


# LanguageTool and Gemini both only need the transcript, so run them
# concurrently instead of back-to-back — this is usually the single biggest
# easy win since Gemini is a network call that can take several seconds.
_review_executor = ThreadPoolExecutor(max_workers=4)


def analyze_audio(audio_path: str) -> dict:
    t_start = time.time()

    with _whisper_semaphore:
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
    gemini_future = _review_executor.submit(
        _run_gemini_review, transcript, wpm, len(fillers), long_
    )
    grammar = lt_future.result()
    # Gemini review is required now (see _run_gemini_review's retry logic) —
    # no timeout-and-skip here. If it ultimately fails after all retries,
    # that exception propagates up and the /api/analyze request fails with
    # a clear error instead of silently shipping a session without it.
    corrected, gemini_mistakes, score, score_reason, criteria, strength, improve = gemini_future.result()
    grammar.extend(gemini_mistakes)
    t_review = time.time()
    print(f"[timing] grammar review (LanguageTool + Gemini, parallel): {t_review - t_whisper:.1f}s")
    print(f"[timing] total: {t_review - t_start:.1f}s")

    return {
        "transcript": transcript,
        "corrected": corrected,
        "score": score,
        "score_reason": score_reason,
        "criteria": criteria,
        "strength": strength,
        "improve": improve,
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
            score=analysis.get("score"),
            score_reason=analysis.get("score_reason"),
            rubric_criteria=json.dumps(analysis.get("criteria", {})),
            strength=analysis.get("strength"),
            improve=analysis.get("improve"),
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

    try:
        # analyze_audio() and save_session() are both blocking (CPU/GPU work,
        # subprocess calls, disk I/O) — running them directly in this async
        # function would freeze FastAPI's single event loop for every other
        # request (including someone just loading their history) until this
        # one finishes. asyncio.to_thread hands the work to a background
        # thread so other requests keep being served concurrently.
        analysis = await asyncio.to_thread(analyze_audio, str(save_path))
    except Exception as e:
        # Most likely cause here is _run_gemini_review exhausting its
        # retries (see GEMINI_MAX_RETRIES) — surfaced as a clean error
        # instead of a raw 500 traceback, since Gemini review is required.
        raise HTTPException(502, f"Analysis failed: {e}")

    session_id = await asyncio.to_thread(
        save_session, student_name, roll_number, grade, section,
        str(save_path), analysis,
    )
    analysis["session_id"] = session_id
    return analysis


@app.get("/api/rubric")
def get_rubric():
    """Serves the same rubric text embedded in RUBRIC_PROMPT, so the
    dashboard's 'how this score is calculated' table is always identical to
    what Gemini is actually scoring against."""
    return {
        "criteria": [
            {
                "key": key,
                "label": crit["label"],
                "bands": [
                    {"value": band, "text": crit["bands"][band]}
                    for band in (2.0, 1.5, 1.0, 0.5)
                ],
            }
            for key, crit in RUBRIC_DEFINITIONS.items()
        ],
        "excluded_note": (
            "Confidence and Body Language is part of the school's full "
            "rubric but requires video (eye contact, gestures, posture), "
            "which this app does not capture. It is excluded here."
        ),
        "scaling_note": (
            "The 4 scored criteria are summed (max 8) and rescaled to a "
            "/10 total, rounded to the nearest 0.5."
        ),
    }


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
                "score": s.score,
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