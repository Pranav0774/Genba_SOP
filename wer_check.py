import re
import jiwer

GROUND_TRUTH_PATH = r"D:\genba-sop\reference\English\Ground_Truth\eng_01_ground_truth.txt"
MACHINE_PATH = r"D:\genba-sop\reference\English\Transcript\eng_01.txt"
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

print(f"Word Error Rate: {error * 100:.2f}%")
print(f"(Target for English videos: <= 12%)")

# Detailed breakdown (substitutions, deletions, insertions)
output = jiwer.process_words(
    reference=ground_truth,
    hypothesis=machine,
    reference_transform=transform,
    hypothesis_transform=transform,
)
print(f"\nSubstitutions: {output.substitutions}")
print(f"Deletions:     {output.deletions}")
print(f"Insertions:    {output.insertions}")
print(f"Hits:          {output.hits}")
