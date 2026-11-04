import base64
import hashlib
import io
import json
import os
import re
import time

import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "qwen2.5vl:7b"
TEXT_MODEL = "qwen2.5:7b-instruct"

MAX_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 900

CANDIDATES_PER_STEP = 8
MAX_IMAGE_WIDTH = 640
TIMING_PENALTY_PER_SECOND = 0.1
SKIP_PENALTY = 3.0
LOW_CONFIDENCE_SCORE = 5.0

DESCRIBE_PROMPT = (
    "This is a frame from a workplace training video. Describe it in ONE short sentence: "
    "the equipment or objects shown and what is being done. If there is any on-screen text or caption, "
    "quote it exactly after the word TEXT:. Be factual and do not guess."
)

TRANSLATE_PROMPT = """Translate this workplace instruction into one short, plain English sentence. Use only English words (no Chinese or Japanese characters). Output only the translation.

{step}
"""

PAIR_PROMPT = """A step of a workplace procedure.
Original wording: {step}
English translation (may be imperfect): {step_en}

Description of one video frame (any on-screen text is quoted and may be in Japanese):
{desc}

Does this frame show this step? Judge by the key objects and actions, and use the original wording if the translation looks wrong.
- 9-10: the description names the key object or action of the step, or on-screen text states the step.
- 4-6: a related setting or equipment, but the key object or action is not mentioned.
- 0-3: unrelated, OR the description names a different object, OR the on-screen text says the opposite of the step (for example text labelled "turn left" when the step says to turn right).
Answer with a single integer from 0 to 10 and nothing else.
"""


def _post(payload, label):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except (requests.exceptions.RequestException, ValueError) as e:
            last_error = e
            print(f"    WARNING: Ollama call failed ({label}) [attempt {attempt}/{MAX_ATTEMPTS}]: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def encode_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.width > MAX_IMAGE_WIDTH:
            height = round(img.height * MAX_IMAGE_WIDTH / img.width)
            img = img.resize((MAX_IMAGE_WIDTH, height))
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def describe_photo(path):
    payload = {
        "model": VISION_MODEL,
        "prompt": DESCRIBE_PROMPT,
        "images": [encode_image(path)],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 100},
    }
    return _post(payload, os.path.basename(path))


def load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path, cache):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def describe_missing(photo_paths, cache_path):
    cache = load_cache(cache_path)
    todo = [p for p in photo_paths if os.path.basename(p) not in cache]
    print(f"Photos to describe: {len(todo)} (already cached: {len(photo_paths) - len(todo)})")
    for i, path in enumerate(todo, 1):
        name = os.path.basename(path)
        started = time.time()
        cache[name] = describe_photo(path)
        save_cache(cache_path, cache)
        print(f"  [{i}/{len(todo)}] {name}: {time.time() - started:.1f}s")
    return cache


def span_cost(photo_t, start, end):
    outside = max(start - photo_t, 0.0, photo_t - end)
    return outside, abs(photo_t - (start + end) / 2)


def _ask_text_model(prompt, num_predict):
    payload = {
        "model": TEXT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 7, "num_predict": num_predict, "num_ctx": 2048},
    }
    return _post(payload, "text model")


def to_english(step_text, score_cache):
    if not re.search(r"[^\x00-\x7f]", step_text):
        return step_text
    key = f"EN||{step_text}"
    if key not in score_cache:
        reply = _ask_text_model(TRANSLATE_PROMPT.format(step=step_text), 60)
        score_cache[key] = reply.split("\n")[0].strip() or step_text
    return score_cache[key]


def score_step(step_text, candidates, descriptions, score_cache):
    # The text model runs at temperature 0 with a fixed seed, so a (step, photo description) pair
    # always scores the same. Caching them lets a run that crashed resume where it stopped.
    step_en = to_english(step_text, score_cache)
    scores = {}
    for k, (path, _) in enumerate(candidates, 1):
        name = os.path.basename(path)
        description = descriptions.get(name, "(no description)")
        key = f"{step_text}||{name}||{hashlib.md5(description.encode('utf-8')).hexdigest()[:8]}"
        if key not in score_cache:
            reply = _ask_text_model(PAIR_PROMPT.format(step=step_text, step_en=step_en, desc=description), 4)
            match = re.search(r"\d+", reply)
            score_cache[key] = max(0.0, min(10.0, float(match.group()))) if match else 0.0
        scores[k] = score_cache[key]
    return step_en, scores


def assign_in_order(step_candidates):
    dp = {-1: (0.0, [])}
    for candidates in step_candidates:
        new_dp = {}
        for last, (total, chain) in dp.items():
            skipped = (total - SKIP_PENALTY, chain + [None])
            if last not in new_dp or skipped[0] > new_dp[last][0]:
                new_dp[last] = skipped
            for global_idx, combined in candidates:
                if global_idx > last:
                    option = (total + combined, chain + [global_idx])
                    if global_idx not in new_dp or option[0] > new_dp[global_idx][0]:
                        new_dp[global_idx] = option
        dp = new_dp
    return max(dp.values(), key=lambda v: v[0])[1]


def match_by_content(steps, photo_ts, segment_ends, cache_path, report_path):
    ordered = sorted(photo_ts, key=lambda pt: (pt[1], pt[0]))
    global_index = {path: i for i, (path, _) in enumerate(ordered)}

    shortlists = []
    for _, _, step_ts in steps:
        if step_ts is None:
            shortlists.append([])
            continue
        step_end = max(step_ts, segment_ends.get(round(step_ts, 1), step_ts))
        ranked = sorted(photo_ts, key=lambda pt: span_cost(pt[1], step_ts, step_end))
        shortlists.append(sorted(ranked[:CANDIDATES_PER_STEP], key=lambda pt: (pt[1], pt[0])))

    unique_paths = []
    for candidates in shortlists:
        for path, _ in candidates:
            if path not in unique_paths:
                unique_paths.append(path)
    descriptions = describe_missing(unique_paths, cache_path)

    score_cache_path = os.path.splitext(cache_path)[0].replace("_photo_descriptions", "") + "_score_cache.json"
    score_cache = load_cache(score_cache_path)
    print(f"Scoring candidates for each step... (cached scores: {len(score_cache)})")
    step_candidates = []
    llm_scores = []
    english_steps = []
    for (step_text, _, step_ts), candidates in zip(steps, shortlists):
        if not candidates:
            step_candidates.append([])
            llm_scores.append({})
            english_steps.append(step_text)
            continue
        step_end = max(step_ts, segment_ends.get(round(step_ts, 1), step_ts))
        try:
            step_en, scores = score_step(step_text, candidates, descriptions, score_cache)
        finally:
            save_cache(score_cache_path, score_cache)
        llm_scores.append(scores)
        english_steps.append(step_en)
        options = []
        for k, (path, photo_t) in enumerate(candidates, 1):
            outside = span_cost(photo_t, step_ts, step_end)[0]
            options.append((global_index[path], scores.get(k, 0.0) - TIMING_PENALTY_PER_SECOND * outside))
        step_candidates.append(options)
        print(f"  [{step_ts:.1f}s] {step_text[:30]}: " + " ".join(f"{k}:{scores.get(k, 0):.0f}" for k in range(1, len(candidates) + 1)))

    chosen = assign_in_order(step_candidates)

    matched = []
    report = ["Photo match report (content-based)", "=" * 60]
    for i, ((step_text, _, step_ts), candidates, global_idx) in enumerate(zip(steps, shortlists, chosen), 1):
        if global_idx is None:
            matched.append(None)
            report.append(f"Step {i} [{step_ts}s]: {step_text}\n  -> no confident content match, falling back to timing\n")
            continue
        path = ordered[global_idx][0]
        matched.append(path)
        k = next(k for k, (p, _) in enumerate(candidates, 1) if p == path)
        score = llm_scores[i - 1].get(k, 0.0)
        flag = "  LOW CONFIDENCE - check this photo" if score < LOW_CONFIDENCE_SCORE else ""
        report.append(
            f"Step {i} [{step_ts}s]: {step_text}\n     (English: {english_steps[i - 1]})\n"
            f"  -> {os.path.basename(path)} (model score {score:.0f}/10){flag}\n"
            f"     {descriptions.get(os.path.basename(path), '')}\n"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Saved photo match report to: {report_path}")
    return matched
