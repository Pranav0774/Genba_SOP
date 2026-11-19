"""Logic behind the Streamlit app (app.py): finding videos, reading and saving the editable SOP data,
cutting frames, running the pipeline scripts and building the Word document. Nothing here imports
Streamlit, so it can be tested on its own."""
import json
import os
import re
import subprocess
import sys
import threading
import time

import cv2
import requests

ROOT = r"D:\genba-sop"
SEPARATOR = " || "
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
EXCLUDE_FRAMES_BEFORE = {"eng_04": 64.5}
NONE_MARKERS = ("記載なし", "None mentioned", "None listed")

LANGUAGE_FOLDERS = {"Japanese": "japanese", "English": "english"}
PREFIXES = {"Japanese": "jap_", "English": "eng_"}


def language_of(video):
    return "Japanese" if video.startswith("jap_") else "English"


def list_videos():
    videos = []
    for language, folder in LANGUAGE_FOLDERS.items():
        directory = os.path.join(ROOT, "videos", folder)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            stem, ext = os.path.splitext(name)
            if ext.lower() == ".mp4" and stem.startswith(PREFIXES[language]) and "_audio" not in stem:
                videos.append(stem)
    return videos


def paths(video):
    language = language_of(video)
    sops = os.path.join(ROOT, "reference", language, "SOPs")
    photos = (os.path.join(ROOT, "photos", video) if language == "Japanese"
              else os.path.join(ROOT, "photos", "English", video))
    transcript_dir = "Transcript_vocab_v3" if language == "Japanese" else "Transcript"
    return {
        "language": language,
        "video": os.path.join(ROOT, "videos", LANGUAGE_FOLDERS[language], f"{video}.mp4"),
        "transcript": os.path.join(ROOT, "reference", language, transcript_dir, f"{video}.txt"),
        "steps": os.path.join(sops, f"{video}_steps.txt"),
        "steps_full": os.path.join(sops, f"{video}_steps_full.txt"),
        "tools": os.path.join(sops, f"{video}_tools_safety.txt"),
        "review": os.path.join(sops, f"{video}_review.json"),
        "output": os.path.join(sops, f"{video}_SOP_reviewed.docx"),
        "output_other": os.path.join(sops, f"{video}_SOP_reviewed_{'en' if language == 'Japanese' else 'ja'}.docx"),
        "log": os.path.join(sops, f"{video}_reviewed_images_log.txt"),
        "frames": os.path.join(photos, "step_frames"),
    }


def artifact_status(video):
    p = paths(video)
    status = {}
    for key, label in (("transcript", "Transcript"), ("steps", "Steps"), ("tools", "Tools and safety"),
                       ("review", "Saved review"), ("output", "Reviewed document")):
        exists = os.path.exists(p[key])
        status[label] = time.strftime("%d %b %H:%M", time.localtime(os.path.getmtime(p[key]))) if exists else None
    return status


def ollama_running():
    try:
        return requests.get(OLLAMA_TAGS_URL, timeout=2).ok
    except requests.exceptions.RequestException:
        return False


def _fix_text(text, language):
    if language == "Japanese":
        from japanese_text import to_japanese_kanji

        return to_japanese_kanji(text)
    return text


def parse_steps_file(path, language):
    """Steps as editable rows. A merged step is stored as 'main || detail || detail'."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().strip().split("\n"):
            m = re.match(r"^\d+\.\s*\[(\d+(?:\.\d+)?)s?\]\s*(.+)$", line.strip())
            if not m:
                continue
            text = m.group(2)
            unclear = "[UNCLEAR]" in text
            parts = [_fix_text(p.strip(), language) for p in text.replace("[UNCLEAR]", "").split("||")]
            rows.append({"keep": True, "ts": float(m.group(1)), "text": parts[0],
                         "details": SEPARATOR.join(parts[1:]), "unclear": unclear})
    return rows


def parse_tools_file(path, language):
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    tools_match = re.search(r"TOOLS AND MATERIALS\s*(.*?)\s*SAFETY POINTS", content, re.DOTALL)
    safety_match = re.search(r"SAFETY POINTS\s*(.*)", content, re.DOTALL)
    def clean(m):
        lines = [_fix_text(l.strip().lstrip("-").strip(), language) for l in (m.group(1).split("\n") if m else [])]
        # "none mentioned" is the model's way of saying the list is empty, not an item
        return [l for l in lines if l and not any(marker in l for marker in NONE_MARKERS)]

    return clean(tools_match), clean(safety_match)


def pipeline_frame_times(video, step_count):
    """The frame times the pipeline chose for each step, read from its match report. Empty if the
    report is missing or was made for a different number of steps."""
    report = os.path.join(os.path.dirname(paths(video)["steps"]), f"{video}_photo_match_report.txt")
    if not os.path.exists(report):
        return []
    with open(report, "r", encoding="utf-8") as f:
        text = f.read()
    step_total = len(re.findall(r"^Step \d+ \[", text, re.MULTILINE))
    times = {int(n): float(t) for n, t in re.findall(r"^Step (\d+) \[[^\n]*\n\s*-> frame at ([\d.]+)s", text, re.MULTILINE)}
    if step_total != step_count:
        return []
    return [times.get(i) for i in range(1, step_count + 1)]


def load_review(video):
    path = paths(video)["review"]
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_review(video, data):
    """Write through a temporary file so an interrupted save never leaves a half-written review."""
    path = paths(video)["review"]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


LAST_VIDEO_PATH = os.path.join(ROOT, ".streamlit", "last_video.txt")


def last_video():
    try:
        with open(LAST_VIDEO_PATH, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def remember_video(video):
    if video and video != last_video():
        os.makedirs(os.path.dirname(LAST_VIDEO_PATH), exist_ok=True)
        with open(LAST_VIDEO_PATH, "w", encoding="utf-8") as f:
            f.write(video)


def initial_review(video, use_full_steps=False):
    """Editable data for a video from the pipeline's output files (nothing from an earlier review)."""
    p = paths(video)
    steps_path = p["steps_full"] if use_full_steps and os.path.exists(p["steps_full"]) else p["steps"]
    steps = parse_steps_file(steps_path, p["language"])
    tools, safety = parse_tools_file(p["tools"], p["language"])
    # start from the frames the pipeline picked; fall back to two seconds after the step starts
    picked = [] if use_full_steps else pipeline_frame_times(video, len(steps))
    for i, step in enumerate(steps):
        step["id"] = i
        step["photo"] = picked[i] if picked and picked[i] is not None else round(step["ts"] + 2.0, 2)
    # steps where the second recogniser disagrees start marked 要確認 (asr_compare.py output, Japanese only)
    for segment in asr_check(video):
        owner = step_at(steps, segment["start"])
        if owner is not None:
            owner["unclear"] = True
    return {"steps": steps, "tools": tools, "safety": safety}


def asr_check(video):
    """Segments asr_compare.py flagged (large-v3 and ReazonSpeech disagree), or [] if it has not been run."""
    path = os.path.join(ROOT, "reference", "Japanese", "ASR_compare", f"{video}_asr_compare.json")
    if language_of(video) != "Japanese" or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [s for s in data["segments"] if s.get("flagged") and s["disagreement"] >= data["flag_threshold"]]


def reversed_meaning(video, items):
    """English only: [(item, transcript sentence)] for items that say the opposite of the speech
    (polarity_check.py). Works on the current, edited text, so it clears once the item is fixed."""
    p = paths(video)
    if p["language"] != "English" or not os.path.exists(p["transcript"]):
        return []
    from polarity_check import conflicts
    with open(p["transcript"], "r", encoding="utf-8") as f:
        return conflicts([i for i in items if i], f.read())


def step_at(steps, t):
    """The step whose speech covers time t: the last step starting at or before it (steps in any order)."""
    earlier = [s for s in steps if s.get("ts") is not None and s["ts"] <= t + 0.5]
    return max(earlier, key=lambda s: s["ts"]) if earlier else None


_captures = {}
_capture_lock = threading.Lock()


def read_frame(video_path, t):
    """The video frame at time t (BGR), or None."""
    with _capture_lock:
        cap = _captures.get(video_path)
        if cap is None:
            cap = _captures[video_path] = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(float(t), 0.0) * 1000)
        ok, frame = cap.read()
    return frame if ok else None


def video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    return seconds


def auto_pick_frames(video, steps):
    """Run the pipeline's frame picker on the current steps; returns one time (or None) per step."""
    from frame_matching import extract_step_frames
    from generate_sop_docx import load_segment_ends

    p = paths(video)
    tuples = [(s["text"], bool(s.get("unclear")), s["ts"]) for s in steps]
    out_dir = os.path.join(p["frames"], "auto")
    report = os.path.join(out_dir, f"{video}_frame_report.txt")
    os.makedirs(out_dir, exist_ok=True)
    _, times = extract_step_frames(
        p["video"], tuples, load_segment_ends(p["transcript"]), out_dir, video, report,
        EXCLUDE_FRAMES_BEFORE.get(video, 0.0), return_times=True,
    )
    return times


def other_language(language):
    return "English" if language == "Japanese" else "Japanese"


def split_details(details):
    return [part.strip() for part in (details or "").split("||") if part.strip()]


def collect_translation_items(steps, tools, safety):
    """Every line that needs translating, as (text, kind): step wording and details are sentences, tool
    names are terms."""
    items = []
    for step in steps:
        items.append((step["text"], "sentence"))
        items.extend((part, "sentence") for part in split_details(step.get("details")))
    items.extend((tool, "term") for tool in tools)
    items.extend((point, "sentence") for point in safety)
    return list(dict.fromkeys(items))


def translated_copy(steps, tools, safety, translations, target):
    """The steps, tools and safety points with their wording replaced by the translations. A line with
    no translation keeps its original wording and is marked for checking, so nothing is left blank."""
    from translate_text import VERIFY_MARK_EN, VERIFY_MARK_JA

    mark = VERIFY_MARK_JA if target == "Japanese" else VERIFY_MARK_EN

    def tr(text):
        if translations.get(text):
            return translations[text]
        return text if mark.strip() in text else text + mark

    new_steps = []
    for step in steps:
        copy = dict(step)
        copy["text"] = tr(step["text"])
        copy["details"] = SEPARATOR.join(tr(part) for part in split_details(step.get("details")))
        new_steps.append(copy)
    return new_steps, [tr(t) for t in tools], [tr(s) for s in safety]


def build_sop(video, steps, tools, safety, language=None, output=None, title=None):
    """Write the Word document from the reviewed data. steps: dicts with text, details, unclear, ts,
    photo (video time in seconds, or None for no photo). language and output default to the video's
    own language and the usual reviewed-document path. Returns the path written."""
    import numpy as np  # noqa: F401  (cv2 needs it)
    from generate_sop_docx import build_document

    p = paths(video)
    language = language or p["language"]
    os.makedirs(p["frames"], exist_ok=True)
    tuples, photo_paths = [], []
    for n, step in enumerate(steps, 1):
        text = step["text"] + (SEPARATOR + step["details"] if step.get("details") else "")
        tuples.append((text, bool(step.get("unclear")), step.get("ts")))
        photo_path = None
        if step.get("photo") is not None:
            frame = read_frame(p["video"], step["photo"])
            if frame is not None:
                photo_path = os.path.join(p["frames"], f"{video}-final-step-{n:03d}.jpg")
                cv2.imwrite(photo_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        photo_paths.append(photo_path)

    output = output or p["output"]
    try:
        build_document(title or video, tuples, tools, safety, photo_paths, output, p["log"], language)
    except PermissionError:
        output = output.replace(".docx", f"_{time.strftime('%H%M%S')}.docx")  # the first file is open in Word
        build_document(title or video, tuples, tools, safety, photo_paths, output, p["log"], language)
    return output


def save_upload(language, filename, data):
    """Save an uploaded video under a name the pipeline recognises (jap_... or eng_...)."""
    stem = re.sub(r"[^a-z0-9]+", "_", os.path.splitext(filename)[0].lower()).strip("_") or "video"
    name = PREFIXES[language] + stem
    folder = os.path.join(ROOT, "videos", LANGUAGE_FOLDERS[language])
    os.makedirs(folder, exist_ok=True)
    candidate, n = name, 2
    while os.path.exists(os.path.join(folder, candidate + ".mp4")):
        candidate = f"{name}_{n}"
        n += 1
    with open(os.path.join(folder, candidate + ".mp4"), "wb") as f:
        f.write(data)
    return candidate


def run_stage(script, video, args=()):
    """Run one pipeline script for a video, yielding its output lines, then the exit code (an int)."""
    env = dict(os.environ)
    env.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [sys.executable, "-u", script, video, *args], cwd=ROOT,  # -u: show the script's output as it runs
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    for line in process.stdout:
        yield line.rstrip("\n")
    yield process.wait()
