import subprocess
import os
from faster_whisper import WhisperModel


VIDEO_PATH = r"D:\genba-sop\videos\english\eng_01.mp4"
AUDIO_PATH = r"D:\genba-sop\videos\english\eng_01_new_audio.wav"
MODEL_SIZE = "small"   # options: tiny, base, small, medium, large-v3
                        # start small — faster to download and test with

# ---- Extract audio using ffmpeg ----
print("Extracting audio from video...")
subprocess.run([
    "ffmpeg",
    "-y",                
    "-i", VIDEO_PATH,
    "-ar", "16000",      
    "-ac", "1",          
    AUDIO_PATH
], check=True)

print(f"Audio extracted to: {AUDIO_PATH}")

# ---- Load faster-whisper model ----
print(f"Loading faster-whisper model ({MODEL_SIZE})...")

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

# ---- Transcribe with VAD enabled ----
print("Transcribing...")
segments, info = model.transcribe(
    AUDIO_PATH,
    vad_filter=True,
    language=None
)

print(f"\nDetected language: {info.language} (confidence: {info.language_probability:.2f})\n")
print("=" * 60)
print("TRANSCRIPT")
print("=" * 60)

for segment in segments:
    print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")

print("=" * 60)
print("\nTranscription Complete.")