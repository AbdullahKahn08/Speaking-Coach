"""
Consolidated local-model benchmark for the rubric-scoring task.

Runs every model in CANDIDATE_MODELS against every transcript in
TEST_CASES, and compares each result to Claude's already-recorded score
for that same transcript - so you get one table at the end instead of
juggling separate files/screenshots per model.

This does NOT touch main.py / main2.py / ollama_main.py / ollama_deepseek_main.py
at all - purely a comparison tool.

Run:
    ollama serve          (if not already running)
    python benchmark_models.py

Models not currently pulled are skipped automatically (reported, not a
crash) - so you can add new tags to CANDIDATE_MODELS ahead of pulling them,
and just re-run once they're ready.
"""

import json
import re
import time
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"

# Add/remove tags here as you pull or drop models - this is the single
# place that controls what gets tested.
CANDIDATE_MODELS = [
    "qwen2.5:7b",
    "qwen2.5:14b",
    "deepseek-r1:14b",
    "qwen3:14b",
    "gpt-oss:20b",
]

CRITERIA_KEYS = ("content_coherence", "fluency", "accuracy_pronunciation", "expression")
VALID_BANDS = {2.0, 1.5, 1.0, 0.5}

# Each test case carries Claude's already-recorded score (from earlier
# testing) so every model's result is automatically compared against it.
TEST_CASES = [
    {
        "name": "Transcript A (air under the knife)",
        "transcript": (
            "have some air under the knife don't rush it take some rest have an open "
            "mind come back give the presentation a deal that is finished with dirt "
            "then let's go to get a photo of the meeting it goes well then you go to "
            "the bathroom see that a good bathtub then it will give you a good treat "
            "that's not that a good meeting"
        ),
        "wpm": 135, "filler_count": 0, "long_pauses": 0,
        "claude_score": 5.0,
        "claude_criteria": {
            "content_coherence": 0.5, "fluency": 1.5,
            "accuracy_pronunciation": 1.0, "expression": 1.0,
        },
    },
    {
        "name": "Transcript B (Tony / Muhammad Abdullah)",
        "transcript": (
            "Hello, my name is Tony. My best friend is Muhammad Abdullah. Muhammad "
            "Abdullah is very smart. He do work. Work good. CEO involved. Me no "
            "involved. Okay. We are going to buy some fruits. Fruits very good. "
            "Fruits good for you. Yes."
        ),
        "wpm": 116, "filler_count": 0, "long_pauses": 2,
        "claude_score": 4.0,
        "claude_criteria": {
            "content_coherence": 0.5, "fluency": 1.0,
            "accuracy_pronunciation": 0.5, "expression": 1.0,
        },
    },
]

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

Return ONLY a JSON object (no prose, no markdown fences) with this exact shape:
{{
  "criteria": {{
    "content_coherence": {{"band": 2.0, "comment": "one formal sentence justifying this band"}},
    "fluency": {{"band": 2.0, "comment": "one formal sentence justifying this band"}},
    "accuracy_pronunciation": {{"band": 2.0, "comment": "one formal sentence justifying this band"}},
    "expression": {{"band": 2.0, "comment": "one formal sentence justifying this band"}}
  }}
}}

Each "band" value must be exactly one of 2.0, 1.5, 1.0, or 0.5 - no other numbers."""

# --- Calibration experiment ---
# Every gap this benchmark has caught so far traces to one of a few
# specific, repeatable mistakes - not vague "the model was too soft":
#   - Qwen3 awarded Fluency 2.0 ("without repetition") on a transcript that
#     repeated "good"/"fruits"/"involved" multiple times - it never
#     actually checked the band's own "without repetition" clause against
#     the text.
#   - Multiple models treated "jumps between disconnected ideas with no
#     logical connectors" as merely "ideas a bit limited" (1.5) rather
#     than a real coherence breakdown (0.5/1.0).
#   - qwen2.5:7b gave Accuracy 2.0 to a grammatically-parseable but
#     semantically nonsensical sentence ("a deal that is finished with
#     dirt") - it checked grammar, not meaning.
# This variant adds explicit, checkable instructions for exactly those
# three failure modes, plus a "verify each clause" self-check and a
# round-down tie-breaker - a direct attempt to close the measured
# leniency gap, not a generic "be stricter" instruction.
CALIBRATION_BLOCK = """

CALIBRATION GUIDANCE - apply each of these before finalizing any band:

- Before awarding Fluency 2.0 or 1.5 (both require "without repetition" or
  only "occasional repetition"), scan the transcript for any word or short
  phrase that appears three or more times. If the same word or phrase
  recurs throughout - not just once incidentally - that is frequent
  repetition and caps the band at 1.0, regardless of pause data alone.

- "On-topic" is not the same as "coherent." A response that jumps between
  disconnected statements with no logical connectors linking them - even
  if every sentence is individually on the general subject - is a
  coherence breakdown (0.5 or 1.0 on Content Relevancy & Coherence), not
  merely "ideas a bit limited" (1.5). Reserve 1.5 for a response that is
  underdeveloped but still follows a logical thread, not one that is
  fragmented or disjointed.

- For Accuracy & Pronunciation, check whether each sentence is
  semantically well-formed, not only whether it is grammatically
  parseable. A sentence that is grammatically simple but does not
  actually convey a sensible meaning (mismatched or nonsensical word
  combinations) is a more serious error than a minor grammar slip, and
  must not receive 2.0 or 1.5.

- Before finalizing any band, re-read that exact band's description one
  clause at a time and verify each individual claim in it is actually
  true of this specific response. If any clause is false, that band is
  wrong - move to the next band down.

- When genuinely torn between two adjacent bands, choose the lower
  (stricter) one. A formal assessment does not round up."""

CALIBRATED_RUBRIC_PROMPT = RUBRIC_PROMPT.replace(
    '("Confidence and Body Language"',
    CALIBRATION_BLOCK.strip("\n") + '\n\n("Confidence and Body Language"',
)

# Which prompt variants to test each model with. "standard" is the
# original rubric prompt (baseline); "calibrated" adds the guidance above.
# Testing both, per model, is what makes this an actual experiment rather
# than a guess about whether the extra instructions help.
PROMPT_VARIANTS = {
    "standard": RUBRIC_PROMPT,
    "calibrated": CALIBRATED_RUBRIC_PROMPT,
}


def _extract_json_object(text: str) -> str:
    """Finds the first complete, balanced {...} object in text - handles
    stray commentary before/after the JSON and braces inside quoted
    strings. Works for both plain instruct output and reasoning-model
    output (once the <think> block is stripped by _strip_thinking)."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _strip_thinking(raw_text: str) -> str:
    """Strips a <think>...</think> block if present (DeepSeek-R1, and
    Qwen3 in its default thinking mode both do this) - a no-op for models
    that don't emit one."""
    match = re.search(r"</think>", raw_text, flags=re.IGNORECASE)
    if match:
        return raw_text[match.end():].strip()
    return raw_text.strip()


def call_ollama(model: str, prompt: str) -> tuple[str, float]:
    """No format="json" constraint - forcing that on a reasoning model
    (DeepSeek-R1, Qwen3's thinking mode) suppresses its actual reasoning
    and produces generic, undifferentiated scores. Letting every model
    think freely and extracting the JSON ourselves afterward works
    uniformly across both plain instruct and reasoning models."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=240) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", ""), time.time() - start


def score_transcript(model: str, case: dict, prompt_template: str):
    """Returns (criteria_dict, total_score, elapsed_sec) or raises."""
    prompt = prompt_template.format(
        transcript=case["transcript"], wpm=case["wpm"],
        filler_count=case["filler_count"], long_pauses=case["long_pauses"],
    )
    raw_text, elapsed = call_ollama(model, prompt)
    raw_text = _strip_thinking(raw_text)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip()).strip()
    raw = _extract_json_object(raw)
    parsed = json.loads(raw)

    criteria = {}
    raw_sum = 0.0
    for key in CRITERIA_KEYS:
        entry = parsed.get("criteria", {}).get(key, {})
        band = entry.get("band")
        if not isinstance(band, (int, float)) or band not in VALID_BANDS:
            band = min(VALID_BANDS, key=lambda v: abs(v - (band or 1.0))) \
                if isinstance(band, (int, float)) else 1.0
        criteria[key] = band
        raw_sum += band

    total = round((raw_sum / 8.0 * 10) * 2) / 2
    return criteria, total, elapsed


def main():
    results = []  # (model, variant, case_name, status, total, gap, elapsed)

    for model in CANDIDATE_MODELS:
        for variant_name, prompt_template in PROMPT_VARIANTS.items():
            for case in TEST_CASES:
                label = f"{model} [{variant_name}] / {case['name']}"
                print(f"Running {label} ...", end=" ", flush=True)
                try:
                    criteria, total, elapsed = score_transcript(model, case, prompt_template)
                    gap = round(total - case["claude_score"], 1)
                    print(f"done ({elapsed:.0f}s) -> {total}/10 (Claude: {case['claude_score']}/10, gap {gap:+.1f})")
                    results.append((model, variant_name, case["name"], "ok", total, gap, elapsed))
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        print(f"SKIPPED - not pulled locally (ollama pull {model})")
                        results.append((model, variant_name, case["name"], "not pulled", None, None, None))
                    else:
                        print(f"FAILED - HTTP {e.code}")
                        results.append((model, variant_name, case["name"], f"http error {e.code}", None, None, None))
                except Exception as e:
                    print(f"FAILED - {e}")
                    results.append((model, variant_name, case["name"], f"error: {e}", None, None, None))

    # --- Summary table ---
    print()
    print("=" * 110)
    print(f"{'Model':<18} {'Prompt':<12} {'Transcript':<10} {'Score':<9} {'Claude':<8} {'Gap':<7} {'Time':<6} Status")
    print("=" * 110)
    for model, variant, case_name, status, total, gap, elapsed in results:
        claude_ref = next(c["claude_score"] for c in TEST_CASES if c["name"] == case_name)
        short_name = "A" if case_name.startswith("Transcript A") else "B"
        if status == "ok":
            print(f"{model:<18} {variant:<12} {short_name:<10} {str(total)+'/10':<9} {str(claude_ref)+'/10':<8} {gap:+.1f}   {elapsed:.0f}s")
        else:
            print(f"{model:<18} {variant:<12} {short_name:<10} {'-':<9} {str(claude_ref)+'/10':<8} {'-':<7} {'-':<6} {status}")

    # --- Standard vs calibrated, per model: does the calibration prompt
    # actually shrink the average gap, or not? This is the real answer to
    # "does better prompting close the gap" - measured, not assumed.
    print()
    print("Average gap vs Claude, standard vs calibrated prompt (0 = perfect match):")
    print(f"  {'Model':<18} {'Standard':<14} {'Calibrated':<14} {'Improvement':<12}")
    for model in CANDIDATE_MODELS:
        row = f"  {model:<18} "
        std_gaps = [g for m, v, _, s, _, g, _ in results if m == model and v == "standard" and s == "ok"]
        cal_gaps = [g for m, v, _, s, _, g, _ in results if m == model and v == "calibrated" and s == "ok"]
        std_avg = sum(std_gaps) / len(std_gaps) if std_gaps else None
        cal_avg = sum(cal_gaps) / len(cal_gaps) if cal_gaps else None
        row += f"{(f'{std_avg:+.2f}' if std_avg is not None else 'n/a'):<14}"
        row += f"{(f'{cal_avg:+.2f}' if cal_avg is not None else 'n/a'):<14}"
        if std_avg is not None and cal_avg is not None:
            # Improvement = how much closer to 0 the gap got (positive = better)
            improvement = abs(std_avg) - abs(cal_avg)
            row += f"{improvement:+.2f}"
        else:
            row += "n/a"
        print(row)


if __name__ == "__main__":
    main()