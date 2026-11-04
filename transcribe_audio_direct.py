import os
from faster_whisper import WhisperModel

VIDEOS = ["jap_01", "jap_02", "jap_03"]
LANGUAGE_CODE = "ja"
MODEL_SIZE = "large-v3"

AUDIO_TEMPLATE = r"D:\genba-sop\videos\japanese\{video}_audio.wav"
VOCAB_PATH = r"D:\genba-sop\reference\Japanese\genba_vocabulary.txt"
OUTPUT_DIR = r"D:\genba-sop\reference\Japanese\Transcript_vocab_v3"

PROMPT_PREFIX = "建設・工場の安全教育ビデオ。用語: "


def build_initial_prompt(vocab_path):
    if not os.path.exists(vocab_path):
        return None
    with open(vocab_path, "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]
    return PROMPT_PREFIX + "、".join(terms)


initial_prompt = build_initial_prompt(VOCAB_PATH)
print(f"Vocabulary hint: {len(initial_prompt) if initial_prompt else 0} characters")

print(f"Loading faster-whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for video in VIDEOS:
    audio_path = AUDIO_TEMPLATE.format(video=video)
    output_path = os.path.join(OUTPUT_DIR, f"{video}.txt")

    print(f"\n{'=' * 60}\nTranscribing {video}...\n{'=' * 60}")
    segments, info = model.transcribe(
        audio_path,
        vad_filter=True,
        language=LANGUAGE_CODE,
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
        repetition_penalty=1.1,
    )
    print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

    full_text = []
    for segment in segments:
        line = f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}"
        print(line)
        full_text.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))
    print(f"Saved transcript to: {output_path}")
