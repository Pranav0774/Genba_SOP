"""Japanese speech-recognition accuracy on Google FLEURS (ja_jp test split), with its supplied transcripts.

This is a separate check from the Genba videos: FLEURS is read Wikipedia sentences recorded fairly cleanly, so
it measures the recogniser itself on general Japanese, not factory audio or Genba vocabulary. It gives a CER
that does not depend on our own ground-truth files.

Engines:
  whisper       faster-whisper large-v3 with the pipeline's settings, without the Genba vocabulary hint
  whisper-hint  the same with the Genba hint (shows whether the hint hurts ordinary speech)
  reazonspeech  ReazonSpeech k2-v2, the second recogniser used by asr_compare.py

Usage:
  python fleurs_check.py                      # 200 sentences, whisper + reazonspeech
  python fleurs_check.py --limit 0            # the whole test split
  python fleurs_check.py --engines whisper,whisper-hint,reazonspeech

The first run downloads the ja_jp test audio (about 450 MB) into hf_cache and unpacks the chosen files into
fleurs\\ja_jp\\. Results are cached per engine, so an interrupted run continues where it stopped.
Writes reference/Japanese/FLEURS/ (per-sentence CSV) and reference/Japanese/Results/fleurs_ja_res.txt.
"""
import argparse
import csv
import json
import os
import random
import statistics
import sys
import tarfile
import time

os.environ.setdefault("HF_HOME", r"D:\genba-sop\hf_cache")
sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import hf_hub_download

from asr_metrics import char_errors, normalize

ROOT = r"D:\genba-sop"
AUDIO_DIR = os.path.join(ROOT, "fleurs", "ja_jp", "test")
OUT_DIR = os.path.join(ROOT, "reference", "Japanese", "FLEURS")
SUMMARY_PATH = os.path.join(ROOT, "reference", "Japanese", "Results", "fleurs_ja_res.txt")
VOCAB_PATH = os.path.join(ROOT, "reference", "Japanese", "genba_vocabulary.txt")
JAPANESE_PROMPT_PREFIX = "建設・工場の安全教育ビデオ。用語: "   # same as transcribe_video.py
SEED = 0
ENGINES = ("whisper", "whisper-hint", "reazonspeech")

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=200, help="number of sentences, 0 = all")
parser.add_argument("--engines", default="whisper,reazonspeech")
args = parser.parse_args()
engines = [e.strip() for e in args.engines.split(",") if e.strip()]
for e in engines:
    if e not in ENGINES:
        sys.exit(f"Unknown engine {e!r}; choose from {', '.join(ENGINES)}")


def load_test_set(limit):
    """One recording per FLEURS sentence, a fixed random sample of `limit` of them."""
    tsv = hf_hub_download("google/fleurs", "data/ja_jp/test.tsv", repo_type="dataset")
    by_sentence = {}
    with open(tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            sentence_id, file_name, raw = row[0], row[1], row[2]
            by_sentence.setdefault(sentence_id, {"id": sentence_id, "file": file_name, "reference": raw,
                                                 "samples": int(row[5])})
    items = sorted(by_sentence.values(), key=lambda x: int(x["id"]))
    if 0 < limit < len(items):
        items = sorted(random.Random(SEED).sample(items, limit), key=lambda x: int(x["id"]))
    return items


def unpack_audio(items):
    wanted = {it["file"] for it in items if not os.path.exists(os.path.join(AUDIO_DIR, it["file"]))}
    if not wanted:
        return
    print("Downloading the FLEURS ja_jp test audio (about 450 MB, only the first time)...")
    archive = hf_hub_download("google/fleurs", "data/ja_jp/audio/test.tar.gz", repo_type="dataset")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    print(f"Unpacking {len(wanted)} recordings...")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = os.path.basename(member.name)
            if member.isfile() and name in wanted:
                with tar.extractfile(member) as src, open(os.path.join(AUDIO_DIR, name), "wb") as dst:
                    dst.write(src.read())
                wanted.discard(name)
                if not wanted:
                    break
    if wanted:
        print(f"Warning: {len(wanted)} recordings were not found in the archive.")


def whisper_engine(with_hint):
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    prompt = None
    if with_hint:
        with open(VOCAB_PATH, encoding="utf-8") as f:
            prompt = JAPANESE_PROMPT_PREFIX + "、".join(line.strip() for line in f if line.strip())

    def run(path):
        segments, _ = model.transcribe(path, language="ja", initial_prompt=prompt, condition_on_previous_text=False,
                                       repetition_penalty=1.1, vad_filter=True)
        return "".join(s.text for s in segments).strip()
    return run


def reazonspeech_engine():
    import reazonspeech_asr
    recognizer = reazonspeech_asr.load()
    return lambda path: reazonspeech_asr.transcribe(recognizer, reazonspeech_asr.read_audio(path))


def run_engine(engine, items):
    cache_path = os.path.join(OUT_DIR, f"hyp_{engine}.json")
    cache = {"texts": {}, "seconds": 0.0, "audio_seconds": 0.0}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    todo = [it for it in items if it["file"] not in cache["texts"]]
    if not todo:
        return cache
    print(f"\n{engine}: {len(todo)} recordings to transcribe ({len(items) - len(todo)} already done)")
    run = reazonspeech_engine() if engine == "reazonspeech" else whisper_engine(engine == "whisper-hint")
    began = time.time()
    for n, it in enumerate(todo, 1):
        t0 = time.time()
        cache["texts"][it["file"]] = run(os.path.join(AUDIO_DIR, it["file"]))
        cache["seconds"] += time.time() - t0
        cache["audio_seconds"] += it["samples"] / 16000
        if n % 10 == 0 or n == len(todo):
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            left = (time.time() - began) / n * (len(todo) - n)
            print(f"  {n}/{len(todo)}  about {left / 60:.0f} min left")
    return cache


def score(engine, items, cache):
    rows, S, D, I, N = [], 0, 0, 0, 0
    for it in items:
        hyp = cache["texts"][it["file"]]
        s, d, i = char_errors(it["reference"], hyp)
        n = len(normalize(it["reference"]))
        S, D, I, N = S + s, D + d, I + i, N + n
        rows.append({"id": it["id"], "file": it["file"], "cer": (s + d + i) / max(n, 1),
                     "reference": it["reference"], "hypothesis": hyp})
    with open(os.path.join(OUT_DIR, f"fleurs_ja_{engine}.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "file", "cer", "reference", "hypothesis"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "cer": f"{r['cer']:.3f}"})
    return {"engine": engine, "cer": (S + D + I) / max(N, 1), "S": S, "D": D, "I": I, "chars": N,
            "median": statistics.median(r["cer"] for r in rows), "rows": rows,
            "rtf": cache["seconds"] / max(cache["audio_seconds"], 1e-9)}


os.makedirs(OUT_DIR, exist_ok=True)
items = load_test_set(args.limit)
unpack_audio(items)
results = [score(e, items, run_engine(e, items)) for e in engines]

audio_minutes = sum(it["samples"] for it in items) / 16000 / 60
lines = [
    "FLEURS Japanese (ja_jp, test split) - character error rate with the supplied transcripts",
    f"Sentences: {len(items)} (one recording each, random sample, seed {SEED})   audio: {audio_minutes:.1f} min",
    "CER ignores punctuation and spaces, after NFKC normalisation (full-width -> half-width).",
    "FLEURS is read general text, not factory audio: this measures the recogniser, not the Genba videos.",
    "",
    f"{'engine':<14} {'CER':>7} {'median':>7} {'S':>6} {'D':>6} {'I':>6} {'ref chars':>10} {'speed':>8}",
]
for r in results:
    lines.append(f"{r['engine']:<14} {r['cer']:7.1%} {r['median']:7.1%} {r['S']:6} {r['D']:6} {r['I']:6} "
                 f"{r['chars']:10} {r['rtf']:6.2f}xRT")
lines.append("(speed = processing time / audio time on this laptop's CPU; below 1 is faster than real time)")
for r in results:
    lines += ["", f"Worst 8 sentences for {r['engine']}:"]
    for row in sorted(r["rows"], key=lambda x: -x["cer"])[:8]:
        lines += [f"  {row['cer']:.0%}  id {row['id']}", f"      ref: {row['reference']}",
                  f"      hyp: {row['hypothesis']}"]

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n" + "\n".join(lines[:7 + len(results)]))
print(f"\nSaved: {SUMMARY_PATH}\nPer-sentence results: {OUT_DIR}")
