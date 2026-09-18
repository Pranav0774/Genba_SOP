# Genba Video → SOP Generator

Turns a smartphone-recorded workplace ("Genba") video into a draft Standard Operating Procedure (SOP): a numbered-steps Word document with embedded photos, a tools/materials list, and safety points, in English and Japanese. The goal is a fast AI-generated *first draft* that a human then reviews and edits — not a fully automated, unreviewed output.

Full spec, KPIs, and scope live in [`Genba_Video_SOP_Generator 1.docx`](Genba_Video_SOP_Generator%201.docx); this README summarizes the pipeline, current status, and roadmap as actually implemented in this repo.

## Pipeline

Each stage is a standalone Python script. There is no unified CLI yet — every script is run manually, in order, with input/output paths hardcoded near the top of the file (edit before running).

| # | Stage | Script | What it does |
|---|-------|--------|---------------|
| 1 | Video → audio | ffmpeg (via `transcribe_test.py`) | `ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav` |
| 2 | Audio → transcript | [`transcribe_audio_direct.py`](transcribe_audio_direct.py), [`transcribe_test.py`](transcribe_test.py) | `faster-whisper` (CPU, int8) with Silero VAD, timestamped `[start -> end] text` segments |
| 3 | Accuracy check | [`wer_check.py`](wer_check.py), [`cer_check.py`](cer_check.py) | Compares transcript to hand-written ground truth — word error rate (English) / character error rate (Japanese) via `jiwer` |
| 4 | Noise robustness set | [`mix_noise_snr.py`](mix_noise_snr.py) | RMS-based synthetic noise mixing at controlled SNR (20/10/5 dB), for testing ASR degradation |
| 5 | Video → candidate photos | PySceneDetect (external CLI, see [`reference/pyscenedetect_log.txt`](reference/pyscenedetect_log.txt)) | `scenedetect -i video detect-adaptive --threshold 8 --min-scene-len 10s save-images` → 3 candidate frames (start/mid/end) per detected scene, plus a scene-timestamp CSV |
| 6 | Photos → descriptions | [`vision_test.py`](vision_test.py), [`vision_test_batch.py`](vision_test_batch.py) | Each candidate photo sent to a local Ollama `qwen2.5vl:7b` vision model for a 1–2 sentence action/tool/safety description |
| 7 | Transcript → steps | [`step_split.py`](step_split.py) | Chunks the transcript and sends it to a local Ollama `qwen2.5:7b-instruct` model, which discards small talk/meta-commentary and keeps only genuine action steps, flagging ambiguous ones `[UNCLEAR]`; a second consolidation pass cleans the combined list |
| 8 | Transcript → tools & safety | [`extract_tools_safety.py`](extract_tools_safety.py) | One Ollama call extracts a "Tools and Materials" list and a "Safety Points" list |
| 9 | Assemble SOP | [`generate_sop_docx.py`](generate_sop_docx.py) | Matches each step to the closest-timestamped candidate photo, builds the final Word doc (`python-docx`) with headings, bullet lists, and numbered steps with embedded photos; unclear steps get a bold **要確認** (verify) marker; also writes a plain-text image-selection log for QA |

A complete, working example of the full pipeline's output is in [`reference/English/SOPs/eng_02_SOP.docx`](reference/English/SOPs/eng_02_SOP.docx) (a 32-step sewing-machine tutorial SOP).

## Tech stack

- **faster-whisper** + **Silero VAD** — speech-to-text (CPU, int8)
- **jiwer** — WER/CER scoring against ground truth
- **PySceneDetect** + **OpenCV** — scene-change detection for candidate photo extraction
- **Ollama** (local, `localhost:11434`) running **Qwen2.5-VL-7B** (vision) and **Qwen2.5-7B-Instruct** (text)
- **python-docx** — Word SOP assembly
- **ffmpeg** — audio extraction/mixing, called only as a subprocess (never linked), to keep its LGPL/GPL licensing isolated from the rest of the MIT/Apache-licensed stack
- **soundfile / numpy** — RMS-based noise mixing

All dependencies are open-source by design (see the license table in the spec doc). There's no `requirements.txt` committed yet — each script's docstring notes its own `pip install` needs.

## Repo layout

```
videos/       source .mp4s + extracted/noisy .wav audio (English & Japanese) — gitignored
photos/       PySceneDetect candidate frames per video, photos/English/eng_0X/ and photos/Japanese/jap_0X/
reference/    transcripts, ground truth, WER/CER results, noise-test results, vision descriptions,
              generated sample SOPs — organized under English/ and Japanese/
noise_samples/  factory noise clips used for synthetic SNR mixing
ollama_models/  local Ollama model blobs — gitignored (~10 GB, never commit)
tools/ffmpeg/   bundled ffmpeg/ffprobe/ffplay binaries — gitignored
venv/, hf_cache/  local virtualenv and HuggingFace cache — gitignored
```

## How to run

1. Activate the local venv; install faster-whisper, jiwer, opencv-python, scenedetect, python-docx, soundfile, numpy, requests.
2. Install ffmpeg (or use the bundled copy in `tools/ffmpeg/`) and [Ollama](https://ollama.com), then pull `qwen2.5vl:7b` and `qwen2.5:7b-instruct`; leave the Ollama server running.
3. Edit the hardcoded video-name/path constants at the top of each script for the video you're processing, then run in order:
   ```
   python transcribe_audio_direct.py
   python wer_check.py            # or cer_check.py for Japanese
   python mix_noise_snr.py        # noise-robustness test set
   scenedetect -i <video> detect-adaptive --threshold 8 --min-scene-len 10s save-images -o <photo_folder>
   python vision_test_batch.py
   python step_split.py
   python extract_tools_safety.py
   python generate_sop_docx.py
   ```

## Current status

**Phase 0 and Phase 1** (environment setup, ASR/VAD in English + Japanese, ground-truth transcripts and WER/CER measurement, synthetic noise test set, scene-detection candidate photos) are complete and were the initial commit.

**Phase 2** (vision descriptions, step splitting, tools/safety extraction, Word SOP assembly) has now been added and proven end-to-end on `eng_02`. It is currently **English-only and single-video-hardcoded** — nothing loops over multiple videos or languages automatically yet.

### Known gaps
- No Streamlit (or any) upload/download UI — every stage is run from the command line
- No automated test suite (pytest)
- No Japanese step-splitting / tools-safety / SOP-assembly path yet (Japanese ASR + ground truth exist, but Phase 2 scripts are English-only)
- No automated candidate → final photo selection; photo curation is manual review of PySceneDetect's output
- No denoising step implemented (DeepFilterNet / noisereduce are in the planned stack but unused — `mix_noise_snr.py` only *adds* synthetic noise for testing, it doesn't remove it)
- No unified CLI — each script requires manually editing hardcoded constants per video

## Roadmap

- **Phase 2 — Steps and Word output** *(this commit)*: vision descriptions, step splitting, tools/safety extraction, Word file with embedded photos, 要確認 flagging. Done when ≥3 videos produce a complete Word SOP (currently 1 of 3).
- **Phase 3 — Japanese output and full noise testing**: extend step-splitting/tools-safety/SOP-assembly to Japanese; run the full 20/10/5 dB noise test matrix across all videos and plot the noise-vs-accuracy curve; measure human edit time on generated SOPs against the 70%-time-saved target.
- **Phase 4 — Demo and handover**: Streamlit upload/download UI, final project report, video-recording guide, demo video, live demo, code handover.

Target PoC KPIs (from the spec doc): English WER ≤12%, Japanese CER ≤15%, ≥80% of true steps found, ≥80% correct photo selection, ≤30 min human edit time (≥70% time saved vs. manual), ≤2% hallucinated content, generation time ≤30 min (Colab) / ≤60 min (laptop CPU).
