import os
import subprocess
import sys

from faster_whisper import WhisperModel

from video_config import video_and_language

FFMPEG = r"D:\genba-sop\tools\ffmpeg\bin\ffmpeg.exe"
VOCAB_PATH = r"D:\genba-sop\reference\Japanese\genba_vocabulary.txt"
JAPANESE_PROMPT_PREFIX = "建設・工場の安全教育ビデオ。用語: "

VIDEO_NAME, LANGUAGE = video_and_language(None)
if VIDEO_NAME is None:
    sys.exit("Usage: python transcribe_video.py <video name, e.g. jap_04 or eng_06> [--force]")

VIDEO_FOLDER = rf"D:\genba-sop\videos\{'japanese' if LANGUAGE == 'Japanese' else 'english'}"
VIDEO_PATH = os.path.join(VIDEO_FOLDER, f"{VIDEO_NAME}.mp4")
AUDIO_PATH = os.path.join(VIDEO_FOLDER, f"{VIDEO_NAME}_audio.wav")
OUTPUT_DIR = (r"D:\genba-sop\reference\Japanese\Transcript_vocab_v3" if LANGUAGE == "Japanese"
              else r"D:\genba-sop\reference\English\Transcript")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}.txt")


def japanese_prompt():
    if not os.path.exists(VOCAB_PATH):
        return None
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]
    return JAPANESE_PROMPT_PREFIX + "、".join(terms)


if os.path.exists(OUTPUT_PATH) and "--force" not in sys.argv:
    sys.exit(f"A transcript already exists: {OUTPUT_PATH}\nRun again with --force to replace it.")

if not os.path.exists(AUDIO_PATH):
    print("Extracting audio from the video...")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", VIDEO_PATH, "-ar", "16000", "-ac", "1", AUDIO_PATH], check=True)

if LANGUAGE == "Japanese":
    model_size, options = "large-v3", dict(
        language="ja", initial_prompt=japanese_prompt(), condition_on_previous_text=False, repetition_penalty=1.1
    )
else:
    model_size, options = "small", dict(language="en")

print(f"Loading faster-whisper model ({model_size})...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")

print(f"Transcribing {VIDEO_NAME}...")
segments, info = model.transcribe(AUDIO_PATH, vad_filter=True, **options)
print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

lines = []
for segment in segments:
    line = f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}"
    print(line)
    lines.append(line)

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Saved transcript to: {OUTPUT_PATH}")
