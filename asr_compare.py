"""Second-opinion check of a Japanese video transcript.

Every segment of the pipeline transcript (large-v3, Transcript_vocab_v3) is transcribed again with ReazonSpeech,
an independent Japanese model. Segments where the two disagree, and segments behind steps marked [UNCLEAR],
are listed with the SOP step they feed, so a reviewer knows what to listen to.

A disagreement is not proof of an error: the two models often write the same word differently (爪 / ツメ), and
both can be wrong in the same way. Treat the list as "listen to this", and the CER figures as provisional
until a spoken-only ground truth is available.

Usage: python asr_compare.py jap_02
Writes reference/Japanese/ASR_compare/<video>_asr_compare.txt and .json, and the ReazonSpeech transcript to
reference/Japanese/Transcript_reazonspeech/<video>.txt (same format as the other transcripts).
"""
import json
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import reazonspeech_asr
from asr_metrics import char_errors, clean_file_text, disagreement, normalize
from video_config import video_and_language

FFMPEG = r"D:\genba-sop\tools\ffmpeg\bin\ffmpeg.exe"
JA = r"D:\genba-sop\reference\Japanese"
DISAGREEMENT_FLAG = 0.35     # segments at or above this are listed for checking
DROPPED_RATIO = 0.5          # ReazonSpeech output shorter than this share of large-v3's = it lost the speech
SEGMENT_MARGIN = 0.3         # seconds of audio added before and after each segment

VIDEO_NAME, LANGUAGE = video_and_language(None)
if VIDEO_NAME is None or LANGUAGE != "Japanese":
    sys.exit("Usage: python asr_compare.py <Japanese video name, e.g. jap_02>")

TRANSCRIPT_PATH = os.path.join(JA, "Transcript_vocab_v3", f"{VIDEO_NAME}.txt")
GROUND_TRUTH_PATH = os.path.join(JA, "Ground_Truth", f"{VIDEO_NAME}_ground_truth.txt")
STEPS_PATH = os.path.join(JA, "SOPs", f"{VIDEO_NAME}_steps.txt")
VIDEO_PATH = rf"D:\genba-sop\videos\japanese\{VIDEO_NAME}.mp4"
AUDIO_PATH = rf"D:\genba-sop\videos\japanese\{VIDEO_NAME}_audio.wav"
OUT_DIR = os.path.join(JA, "ASR_compare")
SECOND_TRANSCRIPT_PATH = os.path.join(JA, "Transcript_reazonspeech", f"{VIDEO_NAME}.txt")


def read_segments(path):
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]\s*(.*)$", line.strip())
            if m:
                segments.append({"start": float(m.group(1)), "end": float(m.group(2)), "whisper": m.group(3)})
    return segments


def read_steps(path):
    steps = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^(\d+)\.\s*\[(\d+(?:\.\d+)?)s?\]\s*(.+)$", line.strip())
                if m:
                    text = m.group(3)
                    steps.append({"number": int(m.group(1)), "ts": float(m.group(2)),
                                  "text": text.replace("[UNCLEAR]", "").split("||")[0].strip(),
                                  "unclear": "[UNCLEAR]" in text})
    return steps


def step_for(segment, steps):
    """The SOP step this segment's speech belongs to: the last step starting at or before it."""
    owner = None
    for step in steps:
        if step["ts"] <= segment["start"] + 0.5:
            owner = step
    return owner


def corpus_cer(reference_text, hypothesis_text):
    s, d, i = char_errors(reference_text, hypothesis_text)
    return (s + d + i) / max(len(normalize(reference_text)), 1), s, d, i


if not os.path.exists(TRANSCRIPT_PATH):
    sys.exit(f"No transcript yet: {TRANSCRIPT_PATH}. Run transcribe_video.py {VIDEO_NAME} first.")
if not os.path.exists(AUDIO_PATH):
    print("Extracting audio from the video...")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", VIDEO_PATH, "-ar", "16000", "-ac", "1", AUDIO_PATH],
                   check=True)

segments = read_segments(TRANSCRIPT_PATH)
steps = read_steps(STEPS_PATH)
audio = reazonspeech_asr.read_audio(AUDIO_PATH)
rate = reazonspeech_asr.SAMPLE_RATE

print("Loading ReazonSpeech (downloads about 160 MB the first time)...")
recognizer = reazonspeech_asr.load()
began = time.time()
for n, seg in enumerate(segments, 1):
    a = max(0, int((seg["start"] - SEGMENT_MARGIN) * rate))
    b = min(len(audio), int((seg["end"] + SEGMENT_MARGIN) * rate))
    seg["reazonspeech"] = reazonspeech_asr.transcribe(recognizer, audio[a:b])
    seg["disagreement"] = round(disagreement(seg["whisper"], seg["reazonspeech"]), 3)
    owner = step_for(seg, steps)
    seg["step"] = owner["number"] if owner else None
    seg["step_unclear"] = bool(owner and owner["unclear"])
    # ReazonSpeech often returns only "はい" for short or quiet segments; that is the second tool missing the
    # speech, not evidence against large-v3, so those segments are listed apart and do not flag a step.
    seg["second_dropped"] = len(normalize(seg["reazonspeech"])) < DROPPED_RATIO * len(normalize(seg["whisper"]))
    seg["flagged"] = (seg["disagreement"] >= DISAGREEMENT_FLAG and not seg["second_dropped"]) or seg["step_unclear"]
    print(f"[{n}/{len(segments)}] {seg['start']:.1f}s  {seg['disagreement']:.0%}  {seg['reazonspeech']}")
elapsed = time.time() - began

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SECOND_TRANSCRIPT_PATH), exist_ok=True)
with open(SECOND_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(f"[{s['start']:.1f}s -> {s['end']:.1f}s] {s['reazonspeech']}" for s in segments))
with open(os.path.join(OUT_DIR, f"{VIDEO_NAME}_asr_compare.json"), "w", encoding="utf-8") as f:
    json.dump({"video": VIDEO_NAME, "flag_threshold": DISAGREEMENT_FLAG, "segments": segments}, f,
              ensure_ascii=False, indent=1)

flagged = [s for s in segments if s["flagged"]]
dropped = [s for s in segments if s["second_dropped"] and not s["flagged"] and s["disagreement"] >= DISAGREEMENT_FLAG]
steps_by_number = {s["number"]: s for s in steps}
flagged_steps = sorted({s["step"] for s in flagged if s["step"] is not None})
whisper_text = "".join(s["whisper"] for s in segments)
second_text = "".join(s["reazonspeech"] for s in segments)

lines = [
    f"ASR second opinion for {VIDEO_NAME}: large-v3 (pipeline transcript) vs ReazonSpeech k2-v2",
    f"Segments: {len(segments)}   flagged: {len(flagged)}   "
    f"(disagreement >= {DISAGREEMENT_FLAG:.0%}, or the segment feeds a step marked [UNCLEAR])   "
    f"not judged (ReazonSpeech lost the speech): {len(dropped)}",
    f"Average disagreement: {sum(s['disagreement'] for s in segments) / max(len(segments), 1):.0%}   "
    f"ReazonSpeech time: {elapsed:.0f} s",
    "",
    "A disagreement means 'listen to this', not 'this is wrong': the models often spell the same word",
    "differently (kanji / kana), and both can mishear the same word.",
    "",
]
if os.path.exists(GROUND_TRUTH_PATH):
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        truth = clean_file_text(f.read())
    lines.append("PROVISIONAL CER against the current ground truth (punctuation ignored). The ground truth still")
    lines.append("contains reviewer notes for jap_02/jap_03, so both figures are too high; compare them with")
    lines.append("each other, not with the 15% target.")
    for label, text in (("large-v3 (pipeline)", whisper_text), ("ReazonSpeech", second_text)):
        rate_, s, d, i = corpus_cer(truth, text)
        lines.append(f"  {label:<20} CER {rate_:6.1%}   S {s}  D {d}  I {i}")
    lines.append("")

if flagged_steps:
    lines.append("SOP steps fed by flagged segments (check these during review):")
    for number in flagged_steps:
        step = steps_by_number[number]
        lines.append(f"  Step {number} [{step['ts']:.1f}s]{' [UNCLEAR]' if step['unclear'] else ''}: {step['text']}")
    lines.append("")

lines.append("FLAGGED SEGMENTS")
for s in flagged:
    where = f"step {s['step']}" if s["step"] is not None else "before the first step"
    reason = "step marked [UNCLEAR]" if s["step_unclear"] and s["disagreement"] < DISAGREEMENT_FLAG else ""
    lines += [
        f"[{s['start']:.1f}s -> {s['end']:.1f}s]  {where}  disagreement {s['disagreement']:.0%}  {reason}".rstrip(),
        f"    large-v3:     {s['whisper']}",
        f"    ReazonSpeech: {s['reazonspeech']}",
    ]

if dropped:
    lines += ["", "NOT JUDGED: ReazonSpeech returned much less text than large-v3, so it gives no second opinion here"]
    for s in dropped:
        lines.append(f"[{s['start']:.1f}s] step {s['step']}  large-v3: {s['whisper']}  |  ReazonSpeech: {s['reazonspeech']}")

report = "\n".join(lines)
report_path = os.path.join(OUT_DIR, f"{VIDEO_NAME}_asr_compare.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print("\n" + "\n".join(lines[:12]))
print(f"\nSaved: {report_path}\nSaved: {SECOND_TRANSCRIPT_PATH}")
