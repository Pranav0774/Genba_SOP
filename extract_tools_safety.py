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
A list of every distinct tool, machine, or material mentioned as being used in the task. Use short, clear names (e.g. "Jigsaw", "2x4 lumber", "Pocket hole jig"). Go through the whole transcript from start to end. Include small items and accessories too (for example pins, clips, markers, tape, a foot pedal, attachments, the manual) and the material being worked on (for example fabric, boards). Do not repeat duplicates. Do NOT list sponsor products, brands or equipment that is only advertised or mentioned in passing (for example a battery pack the host promotes) and is not used to do the task.

2. SAFETY POINTS
A list of any safety-relevant details mentioned or implied — protective equipment, hazards, warnings, or careful-handling instructions. Safety advice is often said casually, not as a warning (for example advice that keeps the work, a tool or a machine secure and stable, keeps hands clear, or controls dust). Include it. If nothing safety-relevant is mentioned, write "None mentioned in transcript" under this section, rather than inventing one.

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
A list of every distinct tool, machine, vehicle, or material mentioned as being used in the task. Write each item in Japanese with a short, clear name (e.g. "フォークリフト", "ジャッキベース", "厚い鉄板"). Go through the whole transcript. Include the machine's parts and controls (levers, brakes, forks, chains, seat belts), safety devices fitted to the machine, communication equipment, documents the worker must carry or check, protective equipment, and the load or pallet being handled. Do NOT list concepts, activities, places, roles or ways of organising work (for example 点検, 地盤, 作業の役割, 指揮命令系統). Do not repeat duplicates.

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


JA_TOOL_VOCAB_PATH = r"D:\genba-sop\reference\Japanese\tool_vocabulary.txt"
JA_ASR_CORRECTIONS_PATH = r"D:\genba-sop\reference\Japanese\asr_corrections.txt"


def _data_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def apply_asr_corrections(text, path=JA_ASR_CORRECTIONS_PATH):
    """Fix known speech-recognition errors (Japanese) before extraction. Returns text and a log."""
    log = []
    for line in _data_lines(path):
        wrong, _, right = line.partition("->")
        wrong, right = wrong.strip(), right.strip()
        count = text.count(wrong)
        if wrong and right and count:
            text = text.replace(wrong, right)
            log.append(f"CORRECTED {count}x: {wrong} -> {right}")
    return text, log


def load_tool_vocabulary(path=JA_TOOL_VOCAB_PATH):
    """[(name, [variants])] from 'Name' or 'Name: variant, variant' lines."""
    entries = []
    for line in _data_lines(path):
        name, sep, variants = line.partition(":")
        name = name.strip()
        entries.append((name, [v.strip() for v in variants.split(",") if v.strip()] if sep else [name]))
    return entries


def vocabulary_tools(transcript, entries):
    """Equipment names that are actually spoken. Longer names are matched first and taken out of the
    text, so ジャッキ inside ジャッキベース, or 鉄板 inside 敷鉄板, is not counted again."""
    text = _normalize(transcript)
    found = []
    variants = sorted(((v, name) for name, vs in entries for v in vs), key=lambda x: -len(x[0]))
    for variant, name in variants:
        key = _normalize(variant)
        if key and key in text:
            text = text.replace(key, "|")
            found.append((name, variant))
    return found


def add_vocabulary_tools(result, transcript):
    """Japanese: add equipment from the vocabulary list that is spoken but missing from the model's
    tools list. Every added item is in the transcript, so none is invented."""
    tools, safety = parse_sections(result)
    have = [_normalize(t) for t in tools]
    added = []
    for name, variant in vocabulary_tools(transcript, load_tool_vocabulary()):
        keys = [_normalize(re.sub(r"（.*?）", "", name)), _normalize(variant)]
        if name in tools or any(k in h or h in k for k in keys for h in have if h):
            continue
        tools.append(name)
        have.extend(keys)
        added.append(f"TOOL added from the vocabulary list: {name} (said as {variant})")
    lines = ["TOOLS AND MATERIALS"] + [f"- {t}" for t in tools]
    lines += ["", "SAFETY POINTS"] + [f"- {s}" for s in (safety or [NO_ITEMS[LANGUAGE]])]
    return "\n".join(lines), added


PART_CHARS = 2500


def split_into_parts(text, size=PART_CHARS):
    """Split at sentence ends into parts of about `size` characters."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > size:
            parts.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        if parts and len(current) < size // 4:
            parts[-1] = f"{parts[-1]} {current}"
        else:
            parts.append(current)
    return parts


def _item_key(item):
    return re.sub(r"[^a-z0-9]", "", item.lower().rstrip("s"))


def merge_results(results):
    """Join the lists from several parts, dropping repeats (same name ignoring case, spaces and a plural s)."""
    tools, safety, seen_tools, seen_safety = [], [], set(), set()
    for result in results:
        part_tools, part_safety = parse_sections(result)
        for item in part_tools:
            if item and _item_key(item) not in seen_tools:
                seen_tools.add(_item_key(item))
                tools.append(item)
        for item in part_safety:
            if item and not any(marker in item for marker in NONE_MARKERS) and _item_key(item) not in seen_safety:
                seen_safety.add(_item_key(item))
                safety.append(item)
    return "\n".join(["TOOLS AND MATERIALS"] + [f"- {t}" for t in tools] + ["", "SAFETY POINTS"]
                     + [f"- {s}" for s in (safety or [NO_ITEMS["English"]])])


def extract_tools_safety(transcript_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        # num_ctx: Ollama's default context is too small for a 10-minute English transcript,
        # so the start of the transcript was cut off before the model saw it.
        "options": {
            "num_predict": 1000,
            "num_ctx": 8192,
            "temperature": 0.1,
            "seed": 7
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

    corrections = []
    if LANGUAGE == "Japanese":
        raw_transcript, corrections = apply_asr_corrections(raw_transcript)
        for c in corrections:
            print(f"  {c}")

    transcript = clean_text(raw_transcript)

    if LANGUAGE == "English":
        # The model misses items in a long transcript (glue said once in a 10-minute video), so English
        # is read in parts of about 2500 characters and the lists are merged.
        parts = split_into_parts(transcript)
        print(f"Sending transcript ({len(transcript)} characters) to {MODEL} in {len(parts)} part(s)...")
        results = []
        for i, part in enumerate(parts, 1):
            print(f"  Part {i}/{len(parts)}...")
            results.append(extract_tools_safety(part))
        result = merge_results(results)
    else:
        print(f"Sending transcript ({len(transcript)} characters) to {MODEL}...")
        result = extract_tools_safety(transcript)

    if LANGUAGE == "Japanese":
        raw_path = OUTPUT_PATH.replace("_tools_safety.txt", "_tools_safety_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(result)
        result, changes = ground_result(result, raw_transcript)
        result, added = add_vocabulary_tools(result, raw_transcript)
        changes = corrections + changes + added
        log_path = OUTPUT_PATH.replace("_tools_safety.txt", "_grounding_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(changes) if changes else "No changes: every tool and safety point is supported.")
        print(f"Checked the lists against the transcript: {len(changes)} change(s)")
        for change in changes:
            print(f"  {change}")
        print(f"Unfiltered model output: {raw_path}\nChange log: {log_path}")
    else:
        # English: flag safety points that say the opposite of the speech ("Do not attend training" when the
        # video says "be sure to attend training"). Flagged, never deleted.
        from polarity_check import conflicts
        tools, safety = parse_sections(result)
        flagged = dict(conflicts(safety, raw_transcript))
        if flagged:
            safety = [f"{s} (please verify: the video seems to say the opposite)" if s in flagged else s for s in safety]
            result = "\n".join(["TOOLS AND MATERIALS"] + [f"- {t}" for t in tools] + ["", "SAFETY POINTS"]
                               + [f"- {s}" for s in safety])
        log_path = OUTPUT_PATH.replace("_tools_safety.txt", "_polarity_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(f"FLAGGED: {item}\nTRANSCRIPT: {sentence}" for item, sentence in flagged.items())
                    or "No safety point contradicts the transcript.")
        print(f"Checked safety points for reversed meaning: {len(flagged)} flagged ({log_path})")

    print("\n" + "=" * 60)
    print("TOOLS AND SAFETY EXTRACTION")
    print("=" * 60)
    print(result)
    print("=" * 60)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nSaved to: {OUTPUT_PATH}")