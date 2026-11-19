import requests
import re
import time
import os
import difflib

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 600

from video_config import video_and_language

VIDEO_NAME, LANGUAGE = video_and_language("jap_03")

TRANSCRIPT_DIR = "Transcript_vocab_v3" if LANGUAGE == "Japanese" else "Transcript"
TRANSCRIPT_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\{TRANSCRIPT_DIR}\{VIDEO_NAME}.txt"
OUTPUT_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_steps.txt"

SEGMENTS_PER_CHUNK = 24 if LANGUAGE == "Japanese" else 8
SNAP_MIN_COVERAGE = 0.5
NUM_PREDICT_CHUNK =800 if LANGUAGE == "Japanese" else 400
EXTRA_OPTIONS = {"num_ctx": 8192, "temperature": 0.1, "seed": 7} if LANGUAGE == "Japanese" else {}

PROMPT_TEMPLATE = """You are converting part of a spoken workplace training transcript into a clean Standard Operating Procedure (SOP).

The transcript below has timestamps in the format [start_seconds -> end_seconds] before each spoken segment.

Your job is to extract ONLY genuine physical actions or instructions someone would need to follow to perform this task. This is NOT a transcript summary or subtitle rewrite - most segments should be DISCARDED.

STRICT RULES - follow these exactly:
1. A step must describe a physical action, decision, or instruction relevant to actually performing the task (e.g. "Drill pilot holes from the outside", "Add glue to each joint").
2. DISCARD segments that are any of the following - do NOT turn them into steps:
   - Small talk, jokes, asides to other people ("can't see the line, can you Russ?")
   - Sponsor reads, product promotions, ads, or brand mentions (e.g. talking about a battery pack's features, sales, giveaways)
   - Personal commentary, opinions, or storytelling not describing an action ("I was not prepared for that", "this hot chocolate is helping")
   - Video/channel meta-commentary ("thanks for watching", "subscribe", "we'll provide plans", "link in description")
   - Restating or previewing what will happen without describing how to do it right now
   - Single-word or filler exclamations ("Mm.", "Flat.", "Okay.")
3. Instructions are often phrased indirectly. KEEP them as steps, for example: "you have to ...", "you also need to ...", "use your ... to ...", "make sure ...", "first ...", "I'll ...", "let's add ...", "next I ...". A choice or setting the worker must make (which size, which setting) is a step too.
   Product recommendations, shopping advice and tips about what to buy are NOT steps.
4. Every step you DO keep must start with its timestamp in square brackets, e.g. "[12.3s] Step text". Do NOT number the steps yourself.
5. Write each kept step as a short imperative instruction, removing filler words.
6. If a kept step is ambiguous or unclear, mark it with "[UNCLEAR]" at the end.

A chunk of 8 transcript segments usually yields between 0 and 4 real steps. Chunks of only talk, sponsor reads or shopping advice yield ZERO steps. But do not skip an instruction just because it is short or phrased as advice: go through every segment and include each distinct action that is said.

Example of correct output format:
[0.2s] Introduce the ultrasound machine being demonstrated
[8.5s] Position the probe against the arm to begin scanning

Transcript segment:
---
{transcript}
---

Output ONLY the list of genuine steps in the exact format shown above (one per line), nothing else. If there are no genuine steps in this segment, output nothing at all. Do not number them, do not add commentary.
"""


PROMPT_TEMPLATE_JA = """You are converting part of a spoken Japanese workplace training transcript into a clean Standard Operating Procedure (SOP).

The transcript below is in Japanese and has timestamps in the format [start_seconds -> end_seconds] before each spoken segment. It was produced by speech recognition, so some words may be misrecognized.

Your job is to extract ONLY genuine physical actions or instructions someone would need to follow to perform this task. This is NOT a transcript summary - most segments should be DISCARDED.

STRICT RULES - follow these exactly:
1. A step must describe a physical action, decision, or instruction relevant to actually performing the task (e.g. "部品を取り付ける", "レバーを引いて固定する").
2. Safety rules, precautions, warnings and prohibitions ARE valid steps, even when stated as a lesson or reason (e.g. "必ず〜する", "〜してはいけない", "電線に注意する", "一つ一つ確認しながら慎重に操作する"). This is often a safety-awareness video, where the rules are the content.
3. DISCARD segments that are any of the following - do NOT turn them into steps:
   - Introductions, quiz questions to the viewer ("どちらが正しいと思うか")
   - Descriptions of what the video will show or what already happened, with no instruction
   - Demonstrations of the WRONG way to do something (keep only the correct method and the rule that follows from it)
   - Questions, small talk, filler, single-word exclamations
   - Meta-commentary about the video ("ご視聴ありがとうございました")
4. DISCARD a segment only when it is clearly not an instruction (a question, greeting, or description of the video). If it states an action, rule, precaution or warning, KEEP it.
5. Every step you keep must start with ONE start time in square brackets, e.g. "[12.3s] ステップ文". Use only the start time. NEVER write a range such as [12.3s -> 15.0s]. Do NOT number the steps yourself.
6. Write each kept step in Japanese as a short imperative instruction, removing filler words. Do not translate to English.
7. Write steps as commands to the worker. Do NOT describe what the people in the video are doing or what the narrator is about to show (e.g. "作業者は〜している", "〜を見てみよう").
8. Quiz videos: if the video presents two options and later reveals which is correct, include ONLY the correct option, written as an instruction. Never include the wrong option or the unsafe example.
9. If a kept step is garbled or looks like a speech-recognition error, keep your best reading and mark it with "[UNCLEAR]" at the end.

You are given a large section of the transcript. It usually yields about 2 to 8 real steps. Use the whole section as context (for example, a later sentence that reveals the correct answer) to decide which method is right. Go through the section from the first segment to the last and include EVERY distinct rule, countermeasure or precaution that is stated. Do not stop early.

Output format (placeholders only - never output these placeholder words; use only what is actually said in the transcript below):
[12.3s] <short Japanese instruction taken from the transcript>
[25.0s] <another short Japanese instruction taken from the transcript> [UNCLEAR]

Transcript segment:
---
{transcript}
---

Output ONLY the list of genuine steps in the exact format shown above (one per line), nothing else. If there are no genuine steps in this segment, output nothing at all. Do not number them, do not add commentary.
"""

if LANGUAGE == "Japanese":
    PROMPT_TEMPLATE = PROMPT_TEMPLATE_JA


def parse_segments(raw_text):
    pattern = r"\[(\d+\.?\d*)s\s*->\s*(\d+\.?\d*)s\]\s*(.*?)(?=\[\d+\.?\d*s\s*->|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    segments = []
    for start, end, text in matches:
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            segments.append((float(start), float(end), text))
    return segments


def chunk_segments(segments, chunk_size, stride=None):
    """Windows of chunk_size segments. With a stride smaller than chunk_size the windows overlap, so an
    instruction at the edge of a window that is mostly talk (for example right after a sponsor read)
    is also seen in a window that is mostly instructions."""
    stride = stride or chunk_size
    for i in range(0, len(segments), stride):
        yield segments[i:i + chunk_size]
        if i + chunk_size >= len(segments):
            break


def drop_overlap_duplicates(steps, max_gap=3.0, min_ratio=0.6):
    """Overlapping windows return the same step twice. Keep the first of two steps that are close in
    time and worded alike."""
    kept = []
    for ts, text in sorted(steps, key=lambda x: x[0]):
        words = text.lower()
        if any(abs(ts - k_ts) <= max_gap and difflib.SequenceMatcher(None, words, k_text.lower()).ratio() >= min_ratio
               for k_ts, k_text in kept):
            continue
        kept.append((ts, text))
    return kept


def segments_to_transcript_text(segments):
    return "\n".join(f"[{s:.1f}s -> {e:.1f}s] {t}" for s, e, t in segments)


def _post_with_retry(payload, label=""):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except (requests.exceptions.RequestException, ValueError) as e:
            last_error = e
            print(f"    WARNING: Ollama call failed{f' ({label})' if label else ''} "
                  f"[attempt {attempt}/{MAX_ATTEMPTS}]: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise last_error


def call_model(transcript_chunk_text, label=""):
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_chunk_text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": NUM_PREDICT_CHUNK, **EXTRA_OPTIONS}
    }
    return _post_with_retry(payload, label=label)


CONSOLIDATE_PROMPT_TEMPLATE = """Below is a draft list of SOP steps extracted from a workplace training video, with timestamps.

Some of these may NOT be genuine physical actions - they could be leftover jokes, sponsor/product mentions, meta-commentary ("thanks for watching"), or vague statements that don't describe a real action.

Review the list and output ONLY the entries that describe a genuine, actionable physical step someone would follow to perform the task. Remove everything else. Keep the original timestamp and wording for the ones you keep - do not rewrite them.

Also remove: entries that only explain a reason or a result and contain no action; filler that announces work without saying what to do (e.g. "Let's begin"); shopping or buying advice (e.g. "Buy a good pair of gloves"); and an entry that repeats the one just before it.

IMPORTANT: testing, trying out, or verifying the finished build (e.g. "try it out with the planer", "test run with the planer", "assign someone to hold the shop vac for dust collection") is a genuine, valuable final step, NOT commentary - keep entries like this. Only remove entries that are actual jokes, sponsor reads, or meta-commentary about the video itself, not steps about validating the finished product.

Draft list:
---
{draft_steps}
---

Output ONLY the kept lines, in the exact same "[X.Xs] text" format, one per line, in their original order. Do not number them, do not add commentary.
"""


CONSOLIDATE_PROMPT_TEMPLATE_JA = """Below is a draft list of SOP steps extracted from a Japanese workplace safety/training video, with timestamps.

Remove ONLY entries that are clearly not instructions: quiz questions, greetings or closing remarks, narration about what the video is about to show, demonstrations of the WRONG method, and exact duplicates.

Keep every entry that states an action, rule, precaution, warning or prohibition, even if it is short or phrased as a lesson. When unsure, KEEP the entry - a human reviews the list afterwards. Keep the original timestamp and wording - do not rewrite or translate anything.

Draft list:
---
{draft_steps}
---

Output ONLY the kept lines, in the exact same "[X.Xs] text" format, one per line, in their original order. Do not number them, do not add commentary.
"""

if LANGUAGE == "Japanese":
    CONSOLIDATE_PROMPT_TEMPLATE = CONSOLIDATE_PROMPT_TEMPLATE_JA


def consolidate_steps(steps):
    if not steps:
        return steps

    draft_text = "\n".join(f"[{ts:.1f}s] {text}" for ts, text in steps)
    prompt = CONSOLIDATE_PROMPT_TEMPLATE.format(draft_steps=draft_text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1500, **EXTRA_OPTIONS}
    }
    try:
        raw_output = _post_with_retry(payload, label="consolidation")
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"    WARNING: consolidation pass failed after {MAX_ATTEMPTS} attempts ({e}). "
              f"Keeping the pre-consolidation draft list instead of losing all progress.")
        return steps
    return parse_step_lines(raw_output)


def _normalize_for_match(text):
    return re.sub(r"[\s、。,.!！?？「」『』（）()・\d\[\]]|UNCLEAR", "", text)


def snap_steps_to_segments(steps, segments):
    """The model often stamps a step with a segment's end time, or a neighbouring segment.
    Move each step to the start of the transcript segment its wording was taken from."""
    normalized = [(s, e, _normalize_for_match(t)) for s, e, t in segments]
    snapped = []
    for ts, text in steps:
        step_norm = _normalize_for_match(text)
        best = None
        for start, _, seg_norm in normalized:
            if not step_norm or not seg_norm:
                continue
            blocks = difflib.SequenceMatcher(None, step_norm, seg_norm, autojunk=False).get_matching_blocks()
            coverage = sum(b.size for b in blocks) / len(step_norm)
            candidate = (coverage, -abs(start - ts), start)
            if best is None or candidate > best:
                best = candidate
        if best is not None and best[0] >= SNAP_MIN_COVERAGE:
            snapped.append((best[2], text))
        else:
            snapped.append((ts, text))
    return sorted(snapped, key=lambda x: x[0])


def parse_step_lines(raw_output):
    lines = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(?:\d+\.\s*)?\[(\d+\.?\d*)s?(?:\s*->\s*\d+\.?\d*s?)?\]\s*(.+)$", line)
        if match:
            ts = float(match.group(1))
            text = match.group(2).strip()
            lines.append((ts, text))
        else:
            print(f"    WARNING: could not parse model line, dropped: {line[:60]}")
    return lines


if __name__ == "__main__":
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        raw_transcript = f.read()

    segments = parse_segments(raw_transcript)
    print(f"Parsed {len(segments)} transcript segments")

    all_steps = []
    # English: windows overlap by half (stride 4 of 8). Japanese keeps its large non-overlapping chunks.
    stride = SEGMENTS_PER_CHUNK // 2 if LANGUAGE == "English" else SEGMENTS_PER_CHUNK
    chunks = list(chunk_segments(segments, SEGMENTS_PER_CHUNK, stride))
    print(f"Processing in {len(chunks)} chunk(s) of up to {SEGMENTS_PER_CHUNK} segments each...")

    for i, chunk in enumerate(chunks, 1):
        chunk_text = segments_to_transcript_text(chunk)
        print(f"  Chunk {i}/{len(chunks)}...")
        try:
            raw_output = call_model(chunk_text, label=f"chunk {i}/{len(chunks)}")
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"    ERROR: chunk {i} failed after {MAX_ATTEMPTS} attempts ({e}). "
                  f"Skipping this chunk and continuing with the rest.")
            continue
        steps = parse_step_lines(raw_output)
        if not steps:
            print(f"    WARNING: chunk {i} produced no parseable steps. Raw output:")
            print(f"    {raw_output[:300]}")
        all_steps.extend(steps)

        partial_path = OUTPUT_PATH + ".partial"
        with open(partial_path, "w", encoding="utf-8") as f:
            f.write("\n".join(f"[{ts:.1f}s] {text}" for ts, text in sorted(all_steps)))

    all_steps.sort(key=lambda x: x[0])
    if LANGUAGE == "English":
        before = len(all_steps)
        all_steps = drop_overlap_duplicates(all_steps)
        print(f"\nRemoved {before - len(all_steps)} duplicate(s) from overlapping windows")
    print(f"\nDraft steps before consolidation: {len(all_steps)}")
    for ts, text in all_steps:
        print(f"  [{ts:.1f}s] {text}")

    print("Running consolidation pass to remove any remaining non-actionable entries...")
    all_steps = consolidate_steps(all_steps)
    print(f"Steps after consolidation: {len(all_steps)}")

    if LANGUAGE == "Japanese":
        all_steps = snap_steps_to_segments(all_steps, segments)
        print("Snapped step timestamps to the start of their source transcript segments.")

    output_lines = []
    for i, (ts, text) in enumerate(all_steps, 1):
        is_unclear = "[UNCLEAR]" in text
        text = text.replace("[UNCLEAR]", "").strip()
        suffix = " [UNCLEAR]" if is_unclear else ""
        output_lines.append(f"{i}. [{ts:.1f}s] {text}{suffix}")

    final_output = "\n".join(output_lines)

    print("\n" + "=" * 60)
    print("GENERATED SOP STEPS")
    print("=" * 60)
    print(final_output)
    print("=" * 60)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(final_output)
    # merge_steps.py always works from this unmerged copy, so merging is repeatable
    with open(OUTPUT_PATH.replace("_steps.txt", "_steps_full.txt"), "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"\nSaved {len(all_steps)} steps to: {OUTPUT_PATH}")

    partial_path = OUTPUT_PATH + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)
