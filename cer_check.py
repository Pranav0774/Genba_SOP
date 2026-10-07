import re
import jiwer

GROUND_TRUTH_PATH = r"D:\genba-sop\reference\Japanese\Ground_Truth\jap_03_ground_truth.txt"
MACHINE_PATH = r"D:\genba-sop\reference\Japanese\Transcript\jap_03.txt"
OUTPUT_PATH = r"D:\genba-sop\reference\Japanese\Results\jap_03_res.txt"
LANGUAGE = "japanese"


def clean_text(text: str) -> str:
    text = re.sub(r"^TIMESTAMPED.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^Source:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^TRANSCRIPT\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Speaker \d+ \(\d{2}:\d{2}\)", "", text)
    text = re.sub(r"\[\d+\.\d+s\s*->\s*\d+\.\d+s\]", "", text)
    text = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth_raw = f.read()

with open(MACHINE_PATH, "r", encoding="utf-8") as f:
    machine_raw = f.read()

ground_truth = clean_text(ground_truth_raw)
machine = clean_text(machine_raw)

if LANGUAGE == "english":
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

elif LANGUAGE == "japanese":
    transform = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])

    error = jiwer.cer(
        reference=ground_truth,
        hypothesis=machine,
        reference_transform=transform,
        hypothesis_transform=transform,
    )
    output = jiwer.process_characters(
        reference=ground_truth,
        hypothesis=machine,
        reference_transform=transform,
        hypothesis_transform=transform,
    )
    result_lines = [
        f"Character Error Rate: {error * 100:.2f}%",
        f"(Target for Japanese videos: <= 15%)",
        "",
        f"Substitutions: {output.substitutions}",
        f"Deletions: {output.deletions}",
        f"Insertions: {output.insertions}",
        f"Hits: {output.hits}",
    ]

else:
    raise ValueError("LANGUAGE must be 'english' or 'japanese'")

result_text = "\n".join(result_lines)
print(result_text)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"\nSaved to: {OUTPUT_PATH}")
