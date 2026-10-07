from faster_whisper import WhisperModel

AUDIO_PATH = r"D:\genba-sop\videos\japanese\jap_03_audio.wav"
MODEL_SIZE = "large-v3"

print(f"Loading faster-whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

print("Transcribing...")
segments, info = model.transcribe(
    AUDIO_PATH,
    vad_filter=True,
    language="ja"
)

print(f"\nDetected language: {info.language} (confidence: {info.language_probability:.2f})\n")
print("=" * 60)
print("TRANSCRIPT")
print("=" * 60)

full_text = []
for segment in segments:
    line = f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}"
    print(line)
    full_text.append(line)

OUTPUT_PATH = r"D:\genba-sop\reference\Japanese\Transcript\jap_03.txt"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(full_text))

print(f"\nSaved transcript to: {OUTPUT_PATH}")