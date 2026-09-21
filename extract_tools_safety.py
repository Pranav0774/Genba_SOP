import requests
import re
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 180

VIDEO_NAME = "eng_05"

TRANSCRIPT_PATH = rf"D:\genba-sop\reference\English\Ground_Truth\{VIDEO_NAME}_ground_truth.txt"
OUTPUT_PATH = rf"D:\genba-sop\reference\English\SOPs\{VIDEO_NAME}_tools_safety.txt"

PROMPT_TEMPLATE = """You are analyzing a spoken workplace training transcript to extract two lists for a Standard Operating Procedure (SOP) document.

Read the transcript below and produce:

1. TOOLS AND MATERIALS
A list of every distinct tool, machine, or material mentioned as being used in the task. Use short, clear names (e.g. "Jigsaw", "2x4 lumber", "Pocket hole jig"). Do not repeat duplicates.

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


def clean_text(text: str) -> str:
    text = re.sub(r"Speaker \d+ \(\d{2}:\d{2}\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_tools_safety(transcript_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 500
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

    print("\n" + "=" * 60)
    print("TOOLS AND SAFETY EXTRACTION")
    print("=" * 60)
    print(result)
    print("=" * 60)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nSaved to: {OUTPUT_PATH}")