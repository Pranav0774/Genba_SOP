# Setup guide

How to install and run the Genba Video → SOP Generator on a new Windows machine. Everything runs
offline after the first setup.

## 1. What you need

| Item | Notes |
|---|---|
| Windows 10 or 11 | Tested on Windows 11 |
| Python 3.14 | Tested on 3.14.3 |
| Git | To clone the repository |
| Disk space | About 15 GB free (models about 10 GB, plus videos and outputs) |
| GPU (optional) | A 4 GB GPU speeds up Ollama. Speech recognition runs on the CPU |

## 2. Get the code

The scripts use absolute paths, so the project must be at `D:\genba-sop`.

```powershell
git clone https://github.com/Pranav0774/Genba_SOP.git D:\genba-sop
cd D:\genba-sop
```

## 3. Python packages

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

To activate the environment in PowerShell, use `venv\Scripts\Activate.ps1` (not `activate.bat`).

## 4. ffmpeg

Download a Windows build of ffmpeg and put `ffmpeg.exe` in `D:\genba-sop\tools\ffmpeg\bin\`.
It is run as a separate program, never linked into the code.

## 5. Ollama (text model)

1. Install Ollama from https://ollama.com and start it.
2. Download the model used for steps, tools and translation:

```powershell
ollama pull qwen2.5:7b-instruct
```

`qwen2.5vl:7b` is only needed for the optional photo-by-content mode and can be skipped.

To keep Ollama models off the C: drive, move `C:\Users\<you>\.ollama\models` to
`D:\genba-sop\ollama_models` and create a link in its place (admin PowerShell):

```powershell
New-Item -ItemType SymbolicLink -Path "C:\Users\<you>\.ollama\models" -Target "D:\genba-sop\ollama_models"
```

## 6. Speech models (downloaded automatically)

Set the model cache to the D: drive once:

```powershell
setx HF_HOME D:\genba-sop\hf_cache
```

These download on first use:

| Model | Used for | Size |
|---|---|---|
| faster-whisper large-v3 | Japanese speech to text | about 3 GB |
| faster-whisper small | English speech to text | about 0.5 GB |
| ReazonSpeech k2-v2 (sherpa-onnx) | Second Japanese recogniser for the accuracy check | about 160 MB |

## 7. Videos

Put videos in `videos\japanese\` or `videos\english\`. The name decides the language:
`jap_....mp4` is Japanese, `eng_....mp4` is English. Videos are not stored in the repository.

## 8. Run the review app

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

The app opens in the browser. Pick a video, run the stages, correct the steps, tools and photos,
then build the Word SOP. Output goes to `reference\<Language>\SOPs\`.

## 9. Run the stages without the app (optional)

```powershell
venv\Scripts\python.exe transcribe_video.py eng_07
venv\Scripts\python.exe step_split.py eng_07
venv\Scripts\python.exe merge_steps.py eng_07
venv\Scripts\python.exe extract_tools_safety.py eng_07
venv\Scripts\python.exe generate_sop_docx.py eng_07
```

## 10. Check that it works

Put a short video (under 2 minutes) in `videos\english\` as `eng_test.mp4` and run all stages from
the app. A 1 to 2 minute clip takes a few minutes. The Word file
`reference\English\SOPs\eng_test_SOP.docx` should have a title, tools, safety points and one
photo per step.

## Problems

| Problem | Fix |
|---|---|
| "model not found" from Ollama | Run `ollama pull qwen2.5:7b-instruct` and make sure Ollama is running |
| ffmpeg not found | Check `D:\genba-sop\tools\ffmpeg\bin\ffmpeg.exe` exists |
| C: drive fills up | Set `HF_HOME` (step 6) and move the Ollama models (step 5) |
| Japanese text shows as boxes in the console | Run `chcp 65001` first, or set `PYTHONIOENCODING=utf-8` |
| Slow first run | The speech models are downloading; later runs are faster |

## Related files

| File | What it is |
|---|---|
| `README.md` | How the pipeline works, the review app, repository layout |
| `requirements.txt` | Python packages with exact versions |
| `app.py`, `app_core.py` | The review app |
| `video_config.py` | Reads the video name and language from the command line |
| `reference/Japanese/genba_vocabulary.txt` | Word hints for Japanese speech recognition |
| `reference/Japanese/tool_vocabulary.txt` | Equipment names used to complete the tools list |
| `reference/Japanese/asr_corrections.txt` | Known speech-recognition errors fixed before extraction |
| `reference/glossary.txt` | Fixed English–Japanese terms for translation |
