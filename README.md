# Genba Video → SOP Generator

Turns a smartphone-recorded workplace ("Genba") video into a draft Standard Operating Procedure (SOP): a Word document with numbered steps, a photo for each step, a tools/materials list and safety points, in Japanese and English. Everything runs locally and offline, so factory video never leaves the machine.

The output is a fast AI-generated *first draft* that a person then reviews and corrects in the review app. It is not meant to be used unreviewed.

Full spec, KPIs and scope are in [`Genba_Video_SOP_Generator 1.docx`](Genba_Video_SOP_Generator%201.docx). This README describes what is implemented in this repo.

## How it works

| # | Stage | Script | What it does |
|---|-------|--------|--------------|
| 1 | Video → transcript | [`transcribe_video.py`](transcribe_video.py) | Extracts audio with ffmpeg and transcribes it with `faster-whisper` on the CPU. Japanese uses `large-v3` with a vocabulary hint ([`reference/Japanese/genba_vocabulary.txt`](reference/Japanese/genba_vocabulary.txt)); English uses `small`. Output is timestamped `[start -> end] text` segments. |
| 2 | Transcript → steps | [`step_split.py`](step_split.py) | A local Ollama model (`qwen2.5:7b-instruct`) keeps only real actions and rules, flags unclear ones `[UNCLEAR]`, and a second pass removes leftovers. Japanese step times are snapped to the transcript segment they came from. |
| 3 | Merge short steps (optional) | [`merge_steps.py`](merge_steps.py) | For demonstration videos, groups consecutive micro-steps of one operation into one step. The model only chooses which steps to group; the wording is never rewritten, and a rule check stops opposite actions being merged. Use `--no-merge` for videos that are a list of independent rules. |
| 4 | Transcript → tools and safety | [`extract_tools_safety.py`](extract_tools_safety.py) | One model call extracts the two lists. For Japanese, tools the transcript never names are removed and weakly supported safety points are marked `（要確認）`; every change is logged. |
| 5 | Photos | [`frame_matching.py`](frame_matching.py) | Cuts each step's photo straight from the video, 1–3 seconds after the step starts, choosing the steadiest, sharpest frame and avoiding dark or blurred ones. No model calls. |
| 6 | Word document | [`generate_sop_docx.py`](generate_sop_docx.py) | Builds the `.docx` with `python-docx`: tools, safety points, then each step in bold above its photo. |
| 7 | Review and translate | [`app.py`](app.py) | Streamlit app: run the stages, correct steps/tools/safety, choose each photo, translate into the other language, and build the final documents. |

Supporting files: [`app_core.py`](app_core.py) (app logic), [`translate_text.py`](translate_text.py) and [`reference/glossary.txt`](reference/glossary.txt) (translation with a domain glossary), [`japanese_text.py`](japanese_text.py) (repairs and detects Chinese characters the model leaks into Japanese), [`video_config.py`](video_config.py) (video name from the command line).

Accuracy and test tooling from Phase 1 is unchanged: [`wer_check.py`](wer_check.py), [`cer_check.py`](cer_check.py), [`mix_noise_snr.py`](mix_noise_snr.py).

## Japanese accuracy checks without a Japanese reviewer

No Japanese reviewer is available yet, so the video CER figures in `reference\Japanese\Results\` are **provisional**. Two checks stand in until a reviewer is found:

- [`asr_compare.py`](asr_compare.py) `jap_02` transcribes every segment again with ReazonSpeech ([`reazonspeech_asr.py`](reazonspeech_asr.py)), an independent Japanese model, and lists the segments where it disagrees with large-v3 together with the SOP step each one feeds. Output: `reference\Japanese\ASR_compare\` and `reference\Japanese\Transcript_reazonspeech\`. A disagreement means "listen to this", not "this is wrong". Places where ReazonSpeech returned almost nothing are listed separately as "not judged". The review app lists the flagged places on the Steps tab, and steps containing them start marked 要確認.
- [`fleurs_check.py`](fleurs_check.py) measures CER on the [Google FLEURS](https://huggingface.co/datasets/google/fleurs) Japanese test split using its supplied transcripts (200 sentences by default, `--limit 0` for all). It measures the recogniser on read general Japanese, not factory audio. Output: `reference\Japanese\Results\fleurs_ja_res.txt`.

## Setup

1. Python 3.14 on Windows, then:
   ```
   python -m venv venv
   venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. Put `ffmpeg.exe` in `tools\ffmpeg\bin\`.
3. Install [Ollama](https://ollama.com), start it, and pull the text model:
   ```
   ollama pull qwen2.5:7b-instruct
   ```
4. Paths in the scripts are absolute (`D:\genba-sop\...`). The project is expected at `D:\genba-sop`.

Videos go in `videos\japanese\` or `videos\english\`. The language is taken from the file name: `jap_...mp4` is Japanese, `eng_...mp4` is English.

## Using the review app

```
venv\Scripts\python.exe -m streamlit run app.py
```

1. **Process** – pick a video (or add one in the sidebar) and run the stages. A 95-second Japanese clip takes about 4 minutes on a laptop CPU with a 4 GB GPU; a full 8-minute video takes much longer, mostly in transcription and step splitting.
2. **Steps** – correct the wording and times, untick steps to leave out, add rows, mark steps 要確認.
3. **Tools & safety** – one item per line. Points marked `（要確認）` need checking.
4. **Photos** – each step starts with the frame the pipeline picked; move it with the time box or ±1 s, or choose "No photo".
5. **Translate** – translates the reviewed text into the other language. Every translation is shown for correction. Extend `reference/glossary.txt` when a term comes out wrong.
6. **Build** – builds the source-language document, the other-language document, or both. Untranslated lines are translated first.

Your edits are saved automatically as `<video>_review.json` as soon as you change something (opening a video without editing writes nothing), and load again next time. The app reopens the video you used last. Reviewed documents are written as `<video>_SOP_reviewed.docx` and `<video>_SOP_reviewed_en.docx` / `_ja.docx` under `reference\<Language>\SOPs\`.

## Running the stages by hand

Each script takes the video name:

```
python transcribe_video.py jap_04
python step_split.py jap_04
python merge_steps.py jap_04            # or: python merge_steps.py jap_04 --no-merge
python extract_tools_safety.py jap_04
python generate_sop_docx.py jap_04      # unreviewed document, straight from the pipeline
```

`transcribe_video.py` refuses to overwrite an existing transcript unless you add `--force`. `step_split.py` gives a different step list on every run, so do not re-run it on a video whose steps have already been reviewed.

## Repo layout

```
videos/         source .mp4s and extracted audio (gitignored)
photos/         scene-detection frames from Phase 1, and step_frames/ cut by the current pipeline
reference/      transcripts, ground truth, WER/CER results, generated SOPs, glossary
                (English/ and Japanese/ subfolders)
noise_samples/  factory noise clips for the synthetic noise test set
ollama_models/, hf_cache/, venv/, tools/ffmpeg/   local models and tools (gitignored)
```

## Current status

**Built and working:** the full pipeline for Japanese and English, the review app, and dual-language output. The app was tested end to end on a new Japanese clip and on all eight project videos.

**Generated SOPs** (pipeline output in `reference\<Language>\SOPs\`):

| Video | State |
|-------|-------|
| eng_02 (sewing machine, 32 steps) | Photos and steps good. |
| eng_03 (workbench, 18 steps) | Photos good. Tools list includes a sponsor product; protective equipment seen in the video is not in the safety list because it is never spoken. |
| eng_04 (work stand, 19 steps) | Photos good; steps 1–2 share a photo; tools list mixes in parts. |
| eng_05 (bottling plant, 20 steps) | A process description from stock footage, not a worker-instruction video. |
| eng_01 (ultrasound, 1 step) | Too little spoken content for a useful SOP. |
| jap_01, jap_02, jap_03 | Generated; photos fit their steps. The text still contains speech-recognition errors and needs review by a Japanese reader before use. |

## Known limits

- **Japanese speech recognition** is the weakest stage. Misheard words (for example ジャッキレース for ジャッキベース) go into the steps, the lists and the translation. The review step is where these are corrected.
- **Translation** is draft quality from a 7B model. Japanese → English is usable; English → Japanese more often needs correcting, and lines the model cannot translate cleanly are left blank for the reviewer.
- **Photos by timing** are right for most steps, but where the picture changes several seconds after the narration (numbered captions in jap_02) the photo can show the previous step.
- **Merging** suits demonstration videos only; on rule-list videos it pairs unrelated rules, so it is off by default in the app.
- The tools/safety grounding filter covers Japanese only. For English, [`polarity_check.py`](polarity_check.py) flags safety points and steps that say the opposite of the speech ("Do not attend training" when the video says "be sure to attend training"); the app shows these in red on the Steps and Tools & safety tabs. It finds reversed instructions only, not other invented content.
- Japanese CER figures for the videos are provisional until a Japanese reviewer checks the ground truth (jap_02 and jap_03 still contain reviewer notes). FLEURS gives an independent CER for the recogniser itself.
- Not implemented: denoising, an automated test suite, a single command that runs every stage.
- The step-splitting model's output varies from run to run.

## PoC targets (from the spec)

English WER ≤12%, Japanese CER ≤15%, ≥80% of true steps found, ≥80% correct photo selection, ≤30 min human edit time (≥70% time saved vs. manual), ≤2% hallucinated content, generation time ≤30 min (Colab) / ≤60 min (laptop CPU). Edit time and step-recall have not been measured yet.
