import os
import re
import shutil
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 600

from video_config import video_and_language

VIDEO_NAME, LANGUAGE = video_and_language("jap_03")

STEPS_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_steps.txt"
FULL_STEPS_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_steps_full.txt"
LOG_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_merge_log.txt"

MAX_GROUP_SIZE = 3
SUB_STEP_SEPARATOR = " || "

# Rule-based guard against merging different operations (Japanese only). The model sometimes ignores
# the instruction not to, so a group that breaks these rules is split back into single steps.
LIFT_WORDS = r"上げ|下げ|下ろ|降ろ|起こ"
RAISE_WORDS = r"上げ"
LOWER_WORDS = r"下げ|下ろ|降ろ"
MOVE_WORDS = r"前進|バック|進め|進み"

PROMPT_TEMPLATE = """Below is a numbered list of steps for a workplace procedure, in the order they happen, each with a timestamp. The list was produced by speech recognition, so some entries are fragments, restatements or slightly garbled.

Your job is to find runs of CONSECUTIVE entries that belong to one single operation, so they can be shown as one step.

Group consecutive entries when they are:
- the same action stated more than once;
- a small check or adjustment that is part of the same operation (for example raising something a few centimetres and then confirming it is level);
- a fragment that only finishes the previous entry (for example an entry ending in "...したら" or "...になりましたら" with no action of its own).

Do NOT group entries that are different operations (for example lifting, driving, lowering, stopping, getting off). Do NOT group entries whose actions are opposites (raise and lower, forward and back). Do NOT group safety rules or prohibitions with anything else. A group has two or three entries, never more. When unsure, leave entries alone.

Output one line for each group, in the form FIRST-LAST (for example the numbers of the first and last entry of a run, joined by a hyphen). Entries that are in no group are not mentioned. Output ONLY these lines, nothing else.

Steps:
---
{steps}
---
"""


def parse_steps(path):
    steps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().strip().split("\n"):
            m = re.match(r"^\d+\.\s*\[(\d+(?:\.\d+)?)s?\]\s*(.+)$", line.strip())
            if m:
                text = m.group(2)
                unclear = "[UNCLEAR]" in text
                steps.append((float(m.group(1)), text.replace("[UNCLEAR]", "").strip(), unclear))
    return steps


def ask_model(prompt, seed):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 400, "num_ctx": 8192, "temperature": 0.0, "seed": seed},
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
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def parse_ranges(raw, n):
    """Return the model's ranges as [(first, last)], or None if they are not valid: ascending, not
    overlapping, two or three entries each, and within the list."""
    ranges = []
    for line in raw.splitlines():
        m = re.match(r"^(\d+)\s*[-–~〜]\s*(\d+)$", line.strip())
        if m:
            ranges.append((int(m.group(1)), int(m.group(2))))
    previous_last = 0
    for first, last in ranges:
        if first <= previous_last or last <= first or last - first + 1 > MAX_GROUP_SIZE or last > n:
            return None
        previous_last = last
    return ranges


def mixes_operations(members):
    """True if the entries contain opposite actions, or both lifting/lowering and driving."""
    if LANGUAGE != "Japanese":
        return False
    texts = [text for _, text, _ in members]
    has = lambda pattern: any(re.search(pattern, t) for t in texts)
    if has(RAISE_WORDS) and has(LOWER_WORDS):
        return True
    return has(LIFT_WORDS) and has(MOVE_WORDS)


def find_groups(steps):
    listing = "\n".join(f"{i}. [{ts:.1f}s] {text}" for i, (ts, text, _) in enumerate(steps, 1))
    prompt = PROMPT_TEMPLATE.format(steps=listing)
    for attempt in range(1, 4):
        ranges = parse_ranges(ask_model(prompt, seed=7 + attempt), len(steps))
        if ranges is not None:
            return ranges
        print(f"    WARNING: the model's grouping was not valid (attempt {attempt}/3); retrying")
    return None


if __name__ == "__main__":
    if not os.path.exists(FULL_STEPS_PATH):
        shutil.copyfile(STEPS_PATH, FULL_STEPS_PATH)
        print(f"Saved the unmerged step list to: {FULL_STEPS_PATH}")
    steps = parse_steps(FULL_STEPS_PATH)
    print(f"Read {len(steps)} steps from {FULL_STEPS_PATH}")

    # for videos whose steps are independent rules, not parts of one operation (e.g. a safety-rules video)
    if "--no-merge" in sys.argv:
        shutil.copyfile(FULL_STEPS_PATH, STEPS_PATH)
        print(f"Merging is off: restored all {len(steps)} steps to {STEPS_PATH}")
        raise SystemExit(0)

    ranges = find_groups(steps)
    if ranges is None:
        print("Could not get a valid grouping from the model. Leaving the step list unmerged.")
        raise SystemExit(1)

    log = [f"Step merge log - {VIDEO_NAME}", "=" * 60]
    group_of = {}
    for first, last in ranges:
        members = steps[first - 1:last]
        if mixes_operations(members):
            log.append(f"Steps {first}-{last}: NOT merged (different operations)")
            log.extend(f"    [{t:.1f}s] {s}" for t, s, _ in members)
            log.append("")
            continue
        group_of[first] = last

    merged = []
    i = 1
    while i <= len(steps):
        last = group_of.get(i, i)
        members = steps[i - 1:last]
        text = SUB_STEP_SEPARATOR.join(s for _, s, _ in members)
        merged.append((members[0][0], text, any(u for _, _, u in members)))
        if last > i:
            log.append(f"Steps {i}-{last} merged into one step:")
            log.extend(f"    [{t:.1f}s] {s}" for t, s, _ in members)
            log.append("")
        i = last + 1

    lines = [f"{n}. [{ts:.1f}s] {text}{' [UNCLEAR]' if unclear else ''}" for n, (ts, text, unclear) in enumerate(merged, 1)]
    with open(STEPS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    print("\n".join(lines))
    print(f"\n{len(steps)} steps merged into {len(merged)}. Saved to: {STEPS_PATH}")
    print(f"Merge log: {LOG_PATH}")
