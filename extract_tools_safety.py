import os
import requests
import re
import time
import difflib

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 600

from video_config import video_and_language

VIDEO_NAME, LANGUAGE = video_and_language("jap_03")

SOURCE_DIR = "Transcript_vocab_v3" if LANGUAGE == "Japanese" else "Ground_Truth"
SOURCE_SUFFIX = "" if LANGUAGE == "Japanese" else "_ground_truth"
TRANSCRIPT_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\{SOURCE_DIR}\{VIDEO_NAME}{SOURCE_SUFFIX}.txt"
if not os.path.exists(TRANSCRIPT_PATH):
    # a new video has no hand-written ground truth, so use its speech-recognition transcript
    TRANSCRIPT_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\Transcript\{VIDEO_NAME}.txt"
OUTPUT_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_tools_safety.txt"

PROMPT_TEMPLATE = """You are analyzing a spoken workplace training transcript to extract two lists for a Standard Operating Procedure (SOP) document.

Read the transcript below and produce:

1. TOOLS AND MATERIALS
A list of every distinct tool, machine, or material mentioned as being used in the task. Use short, clear names (e.g. "Jigsaw", "2x4 lumber", "Pocket hole jig"). Do not repeat duplicates. Do NOT list sponsor products, brands or equipment that is only advertised or mentioned in passing (for example a battery pack the host promotes) and is not used to do the task.

2. SAFETY POINTS
A list of any safety-relevant details mentioned or implied — protective equipment, hazards, warnings, or careful-handling instructions. If nothing safety-relevant is mentioned, write "None mentioned in transcript" under this section, rather than inventing one.

Transcript:
---
{transcript}
---

Output in exactly this format, nothing else:

TOOLS AND MATERIALS
- item
- item

SAFETY POINTS
- item
- item
"""


PROMPT_TEMPLATE_JA = """You are analyzing a spoken Japanese workplace training transcript (produced by speech recognition, so some words may be misrecognized) to extract two lists for a Standard Operating Procedure (SOP) document.

Read the transcript below and produce:

1. TOOLS AND MATERIALS
A list of every distinct tool, machine, vehicle, or material mentioned as being used in the task. Write each item in Japanese with a short, clear name (e.g. "フォークリフト", "ジャッキベース", "厚い鉄板"). List only physical objects. Do NOT list concepts, activities, roles or systems (for example 点検, 地盤, 作業の役割, 指揮命令系統). Do not repeat duplicates.

2. SAFETY POINTS
A list of safety-relevant details mentioned or implied - protective equipment, hazards, warnings, prohibitions, or careful-handling instructions. Write each point in Japanese as a short sentence. If nothing safety-relevant is mentioned, write "- 記載なし" rather than inventing one.

Transcript:
---
{transcript}
---

Output in exactly this format, nothing else. Keep the two English section headings exactly as written below, but write the items in Japanese:

TOOLS AND MATERIALS
- item
- item

SAFETY POINTS
- item
- item
"""

if LANGUAGE == "Japanese":
    PROMPT_TEMPLATE = PROMPT_TEMPLATE_JA


def clean_text(text: str) -> str:
    text = re.sub(r"Speaker \d+ \(\d{2}:\d{2}\)", "", text)
    text = re.sub(r"\[\d+\.\d+s\s*->\s*\d+\.\d+s\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


NO_ITEMS = {"English": "None mentioned in transcript", "Japanese": "記載なし"}
NONE_MARKERS = ("None mentioned", "記載なし")
SAFETY_MIN_COVERAGE = 0.5
SEGMENT_WINDOW = 3

_TIMESTAMP_MARKERS = r"Speaker \d+ \(\d{2}:\d{2}\)|\[\d+\.\d+s\s*->\s*\d+\.\d+s\]"


def _normalize(text):
    return re.sub(r"[\s、。,.!！?？「」『』（）()・\[\]\-–—:：]", "", text).lower()


def split_segments(raw_transcript):
    """The transcript's spoken segments, normalized for comparison."""
    parts = [_normalize(p) for p in re.split(_TIMESTAMP_MARKERS, raw_transcript)]
    return [p for p in parts if p]


def tool_is_grounded(item, transcript_normalized):
    """A tool or material must be named in the transcript (Japanese only). A short item must appear
    exactly. A longer one may be spelled a little differently by the speech recognition, so any three
    characters of it appearing is enough (for example フォークリフト when only リフトレバー is spoken)."""
    item = _normalize(item)
    if item in transcript_normalized:
        return True
    return len(item) > 3 and any(item[i:i + 3] in transcript_normalized for i in range(len(item) - 2))


def sentence_coverage(sentence, segments):
    """How much of the sentence is wording found together in the transcript, in 0..1. Uses runs of
    two or more characters, in a window of a few neighbouring segments, so a safety point that is a
    paraphrase of the speech still scores high and an invented one does not."""
    target = _normalize(sentence)
    if not target:
        return 0.0
    best = 0.0
    for i in range(len(segments)):
        window = "".join(segments[i:i + SEGMENT_WINDOW])
        blocks = difflib.SequenceMatcher(None, target, window, autojunk=False).get_matching_blocks()
        best = max(best, sum(b.size for b in blocks if b.size >= 2) / len(target))
    return best


def parse_sections(result):
    tools_match = re.search(r"TOOLS AND MATERIALS\s*(.*?)\s*SAFETY POINTS", result, re.DOTALL)
    safety_match = re.search(r"SAFETY POINTS\s*(.*)", result, re.DOTALL)
    split = lambda m: [l.strip().lstrip("-").strip() for l in m.group(1).split("\n") if l.strip().lstrip("-").strip()] if m else []
    return split(tools_match), split(safety_match)


def ground_result(result, raw_transcript):
    """Japanese only. Drop tools the transcript never names (the model invents some), and mark safety
    points the transcript only weakly supports with 要確認 instead of deleting them: wrongly deleting
    a real safety rule is worse than asking a reviewer to look at one. Returns the cleaned text in
    the usual format and a log of what was changed."""
    tools, safety = parse_sections(result)
    segments = split_segments(raw_transcript)
    transcript_normalized = "".join(segments)

    changes = []
    kept_tools = []
    for item in tools:
        if tool_is_grounded(item, transcript_normalized):
            kept_tools.append(item)
        else:
            changes.append(f"TOOL removed, not in the transcript: {item}")
    kept_safety = []
    for item in safety:
        if any(marker in item for marker in NONE_MARKERS):
            continue
        coverage = sentence_coverage(item, segments)
        if coverage >= SAFETY_MIN_COVERAGE:
            kept_safety.append(item)
        else:
            kept_safety.append(f"{item}（要確認）")
            changes.append(f"SAFETY marked 要確認, weak support in the transcript ({coverage:.0%}): {item}")

    lines = ["TOOLS AND MATERIALS"] + [f"- {t}" for t in kept_tools]
    lines += ["", "SAFETY POINTS"] + [f"- {s}" for s in (kept_safety or [NO_ITEMS[LANGUAGE]])]
    return "\n".join(lines), changes


def extract_tools_safety(transcript_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 1000
        }
    }

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except (requests.exceptions.RequestException, ValueError) as e:
            last_error = e
            print(f"    WARNING: Ollama call failed [attempt {attempt}/{MAX_ATTEMPTS}]: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise last_error


if __name__ == "__main__":
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        raw_transcript = f.read()

    transcript = clean_text(raw_transcript)

    print(f"Sending transcript ({len(transcript)} characters) to {MODEL}...")
    result = extract_tools_safety(transcript)

    if LANGUAGE == "Japanese":
        raw_path = OUTPUT_PATH.replace("_tools_safety.txt", "_tools_safety_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(result)
        result, changes = ground_result(result, raw_transcript)
        log_path = OUTPUT_PATH.replace("_tools_safety.txt", "_grounding_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(changes) if changes else "No changes: every tool and safety point is supported.")
        print(f"Checked the lists against the transcript: {len(changes)} change(s)")
        for change in changes:
            print(f"  {change}")
        print(f"Unfiltered model output: {raw_path}\nChange log: {log_path}")

    print("\n" + "=" * 60)
    print("TOOLS AND SAFETY EXTRACTION")
    print("=" * 60)
    print(result)
    print("=" * 60)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nSaved to: {OUTPUT_PATH}")