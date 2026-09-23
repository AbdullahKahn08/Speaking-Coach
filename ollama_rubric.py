"""
Standalone test: does Ollama (running our rubric prompt) return valid,
usable JSON when we force JSON mode via the API - instead of the free-form
markdown `ollama run` gave us in interactive chat?

This does NOT touch main.py / main2.py at all - it's a throwaway script to
answer one question before we bother wiring anything into the real app.

Run:
    ollama serve          (if not already running in the background)
    python test_ollama_rubric.py

Requires nothing beyond the Python standard library.
"""

import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"   # change to "llama3.1:8b" to test the other candidate

CRITERIA_KEYS = ("content_coherence", "fluency", "accuracy_pronunciation", "expression")
VALID_BANDS = {2.0, 1.5, 1.0, 0.5}

# Same transcript + metrics you tested by hand, so results are comparable
# apples-to-apples against the Claude output you already have.
TRANSCRIPT = (
    "have some air under the knife don't rush it take some rest have an open "
    "mind come back give the presentation a deal that is finished with dirt "
    "then let's go to get a photo of the meeting it goes well then you go to "
    "the bathroom see that a good bathtub then it will give you a good treat "
    "that's not that a good meeting"
)
WPM = 135
FILLER_COUNT = 0
LONG_PAUSES = 0

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

1. CONTENT RELEVANCY AND COHERENCE
  2.0 - Maintains relevance throughout without deviating; gives diverse ideas; develops the topic coherently with a proper start, middle and end.
  1.5 - Content is on-topic but ideas are a bit limited, with adequate coherence.
  1.0 - Misses some points and doesn't speak for the given time; some breakdowns in coherence.
  0.5 - Insufficient or totally deviated content; breakdowns in coherence.

2. FLUENCY
  2.0 - Speaks fluently with natural pauses and conversational fillers ("you know", "I mean", "well", "basically"), without repetition or self-correction.
  1.5 - Speaks fluently with little pauses ("um", "uh", "mm-hmm"), showing little hesitation, occasional repetition and self-correction.
  1.0 - Cannot respond without noticeable pauses; may speak slowly, with frequent repetition and self-correction.
  0.5 - Speaks with long pauses, repetitions and hesitation.

3. ACCURACY AND PRONUNCIATION
  2.0 - Produces consistently accurate grammatical structures apart from occasional slips; uses a full range of pronunciation features with precision; effortless to understand.
  1.5 - Produces a majority of error-free sentences with only occasional basic errors; easy to understand with minimal pronunciation errors.
  1.0 - May make frequent grammar mistakes though these rarely cause comprehension problems; can generally be understood, though mispronunciation of individual words reduces clarity at times.
  0.5 - Errors are frequent and may lead to misunderstanding; mispronunciations are frequent and cause difficulty for the listener.

4. EXPRESSION (VOCABULARY AND COHESION)
  2.0 - Uses vocabulary with full flexibility and precision; doesn't repeat the same words/structures; uses a variety of words and structures; uses cohesive devices to connect sentences.
  1.5 - Has a wide enough vocabulary to discuss the topic at length and make meaning clear, despite occasional inappropriate word choices; little use of cohesive devices.
  1.0 - Uses vocabulary with limited flexibility and hardly any cohesive devices.
  0.5 - Makes frequent errors in word choice; gives only simple responses; frequently unable to convey the basic message.

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


def call_ollama(prompt: str, model: str) -> tuple[str, float]:
    """POSTs to Ollama's /api/generate with format="json" to FORCE valid
    JSON output - this is the key difference from `ollama run`'s free-form
    chat mode, which just wrote its own markdown rubric instead of our
    schema."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "format": "json",   # <-- forces the model to emit valid JSON
        "stream": False,
        "options": {"temperature": 0.2},  # lower = more consistent band scoring
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - start
    return body.get("response", ""), elapsed


def validate(parsed: dict) -> list[str]:
    """Checks the parsed JSON actually matches the shape main.py expects.
    Returns a list of problems found (empty list = looks good)."""
    problems = []

    criteria = parsed.get("criteria", {})
    for key in CRITERIA_KEYS:
        entry = criteria.get(key)
        if not entry:
            problems.append(f"Missing criterion: {key}")
            continue
        band = entry.get("band")
        if band not in VALID_BANDS:
            problems.append(f"{key}: band {band!r} is not one of {VALID_BANDS}")
        if not entry.get("comment"):
            problems.append(f"{key}: missing comment")

    for field in ("score_reason", "strength", "improve", "corrected"):
        if not parsed.get(field):
            problems.append(f"Missing or empty field: {field}")

    mistakes = parsed.get("mistakes")
    if mistakes is None:
        problems.append("Missing 'mistakes' field entirely")
    elif not isinstance(mistakes, list):
        problems.append("'mistakes' is not a list")
    else:
        for i, m in enumerate(mistakes):
            for f in ("rule_id", "wrong", "correction", "message"):
                if not m.get(f):
                    problems.append(f"mistakes[{i}]: missing '{f}'")

    return problems


def main():
    prompt = RUBRIC_PROMPT.format(
        transcript=TRANSCRIPT, wpm=WPM,
        filler_count=FILLER_COUNT, long_pauses=LONG_PAUSES,
    )

    print(f"Calling Ollama ({MODEL}) with format=json ...")
    raw_text, elapsed = call_ollama(prompt, MODEL)
    print(f"Response took {elapsed:.1f}s\n")

    print("=" * 70)
    print("RAW MODEL OUTPUT")
    print("=" * 70)
    print(raw_text)
    print()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("=" * 70)
        print("RESULT: FAILED - not valid JSON")
        print("=" * 70)
        print(f"Parse error: {e}")
        return

    problems = validate(parsed)

    print("=" * 70)
    if problems:
        print(f"RESULT: JSON parsed, but {len(problems)} problem(s) found")
        print("=" * 70)
        for p in problems:
            print(f"  - {p}")
    else:
        print("RESULT: PASSED - valid JSON matching the expected schema")
        print("=" * 70)
        raw_sum = sum(parsed["criteria"][k]["band"] for k in CRITERIA_KEYS)
        score = round((raw_sum / 8.0 * 10) * 2) / 2
        print(f"\nComputed score: {score}/10")
        for k in CRITERIA_KEYS:
            c = parsed["criteria"][k]
            print(f"  {k}: {c['band']} - {c['comment']}")
        print(f"\nCorrected: {parsed['corrected']}")
        print(f"\nStrength: {parsed['strength']}")
        print(f"Improve:  {parsed['improve']}")
        print(f"\nMistakes found: {len(parsed.get('mistakes', []))}")


if __name__ == "__main__":
    main()