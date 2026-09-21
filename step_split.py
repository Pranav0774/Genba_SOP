import requests
import re
import time
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 180

VIDEO_NAME = "eng_05"

TRANSCRIPT_PATH = rf"D:\genba-sop\reference\English\Transcript\{VIDEO_NAME}.txt"
OUTPUT_PATH = rf"D:\genba-sop\reference\English\SOPs\{VIDEO_NAME}_steps.txt"

SEGMENTS_PER_CHUNK = 8

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
3. If in doubt whether something is a real actionable step, DISCARD it. A shorter, cleaner list is much better than an overly long one.
4. Every step you DO keep must start with its timestamp in square brackets, e.g. "[12.3s] Step text". Do NOT number the steps yourself.
5. Write each kept step as a short imperative instruction, removing filler words.
6. If a kept step is ambiguous or unclear, mark it with "[UNCLEAR]" at the end.

A typical chunk of 8 transcript segments usually yields somewhere between 0 and 4 real steps - many chunks may yield ZERO steps if nothing actionable was said in that window. That is expected and correct.

Example of correct output format:
[0.2s] Introduce the ultrasound machine being demonstrated
[8.5s] Position the probe against the arm to begin scanning

Transcript segment:
---
{transcript}
---

Output ONLY the list of genuine steps in the exact format shown above (one per line), nothing else. If there are no genuine steps in this segment, output nothing at all. Do not number them, do not add commentary.
"""


def parse_segments(raw_text):
    pattern = r"\[(\d+\.?\d*)s\s*->\s*(\d+\.?\d*)s\]\s*(.*?)(?=\[\d+\.?\d*s\s*->|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    segments = []
    for start, end, text in matches:
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            segments.append((float(start), float(end), text))
    return segments


def chunk_segments(segments, chunk_size):
    for i in range(0, len(segments), chunk_size):
        yield segments[i:i + chunk_size]


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
        "options": {"num_predict": 400}
    }
    return _post_with_retry(payload, label=label)


CONSOLIDATE_PROMPT_TEMPLATE = """Below is a draft list of SOP steps extracted from a workplace training video, with timestamps.

Some of these may NOT be genuine physical actions - they could be leftover jokes, sponsor/product mentions, meta-commentary ("thanks for watching"), or vague statements that don't describe a real action.

Review the list and output ONLY the entries that describe a genuine, actionable physical step someone would follow to perform the task. Remove everything else. Keep the original timestamp and wording for the ones you keep - do not rewrite them.

IMPORTANT: testing, trying out, or verifying the finished build (e.g. "try it out with the planer", "test run with the planer", "assign someone to hold the shop vac for dust collection") is a genuine, valuable final step, NOT commentary - keep entries like this. Only remove entries that are actual jokes, sponsor reads, or meta-commentary about the video itself, not steps about validating the finished product.

Draft list:
---
{draft_steps}
---

Output ONLY the kept lines, in the exact same "[X.Xs] text" format, one per line, in their original order. Do not number them, do not add commentary.
"""


def consolidate_steps(steps):
    if not steps:
        return steps

    draft_text = "\n".join(f"[{ts:.1f}s] {text}" for ts, text in steps)
    prompt = CONSOLIDATE_PROMPT_TEMPLATE.format(draft_steps=draft_text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1200}
    }
    try:
        raw_output = _post_with_retry(payload, label="consolidation")
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"    WARNING: consolidation pass failed after {MAX_ATTEMPTS} attempts ({e}). "
              f"Keeping the pre-consolidation draft list instead of losing all progress.")
        return steps
    return parse_step_lines(raw_output)


def parse_step_lines(raw_output):
    lines = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(?:\d+\.\s*)?\[(\d+\.?\d*)s\]\s*(.+)$", line)
        if match:
            ts = float(match.group(1))
            text = match.group(2).strip()
            lines.append((ts, text))
    return lines


if __name__ == "__main__":
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        raw_transcript = f.read()

    segments = parse_segments(raw_transcript)
    print(f"Parsed {len(segments)} transcript segments")

    all_steps = []
    chunks = list(chunk_segments(segments, SEGMENTS_PER_CHUNK))
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
    print(f"\nDraft steps before consolidation: {len(all_steps)}")

    print("Running consolidation pass to remove any remaining non-actionable entries...")
    all_steps = consolidate_steps(all_steps)
    print(f"Steps after consolidation: {len(all_steps)}")

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

    print(f"\nSaved {len(all_steps)} steps to: {OUTPUT_PATH}")

    partial_path = OUTPUT_PATH + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)
