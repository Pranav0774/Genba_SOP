import re
import jiwer

GROUND_TRUTH_PATH = r"D:\genba-sop\reference\English\Ground_Truth\eng_03_ground_truth.txt"
MACHINE_PATH = r"D:\genba-sop\reference\English\Noise_added\eng_3\eng_03_5db.txt"
OUTPUT_PATH = r"D:\genba-sop\reference\English\Noise_added\eng_3\eng_03_5db_result.txt"
LANGUAGE = "english"


def clean_text(text: str) -> str:
    text = re.sub(r"Speaker \d+ \(\d{2}:\d{2}\)", "", text)
    text = re.sub(r"\[\d+\.\d+s\s*->\s*\d+\.\d+s\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth_raw = f.read()

with open(MACHINE_PATH, "r", encoding="utf-8") as f:
    machine_raw = f.read()

ground_truth = clean_text(ground_truth_raw)
machine = clean_text(machine_raw)

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

error = jiwer.wer(
    reference=ground_truth,
    hypothesis=machine,
    reference_transform=transform,
    hypothesis_transform=transform,
)

output = jiwer.process_words(
    reference=ground_truth,
    hypothesis=machine,
    reference_transform=transform,
    hypothesis_transform=transform,
)

result_lines = [
    f"Word Error Rate: {error * 100:.2f}%",
    f"(Target for English videos: <= 12%)",
    "",
    f"Substitutions: {output.substitutions}",
    f"Deletions: {output.deletions}",
    f"Insertions: {output.insertions}",
    f"Hits: {output.hits}",
]
result_text = "\n".join(result_lines)

print(result_text)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"\nSaved to: {OUTPUT_PATH}")
