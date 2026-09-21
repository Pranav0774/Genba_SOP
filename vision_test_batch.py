import requests
import base64
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:7b"

PHOTO_FOLDER = r"D:\genba-sop\photos\English\eng_03"
OUTPUT_PATH = r"D:\genba-sop\reference\English\Vision_des\vision_descriptions_eng_03.txt"

PROMPT = (
    "You are looking at a photo from a workplace training video. "
    "Describe in one or two clear sentences what action or step is being shown. "
    "Focus on what the person is doing, what tool or object they are using, "
    "and any safety-relevant detail visible (e.g. gloves, protective gear, hazards). "
    "Be concise and factual."
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(image_path):
    image_b64 = encode_image(image_path)
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json().get("response", "").strip()


if __name__ == "__main__":
    image_files = sorted([
        f for f in os.listdir(PHOTO_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Found {len(image_files)} images in {PHOTO_FOLDER}")

    results = []
    for i, filename in enumerate(image_files, 1):
        full_path = os.path.join(PHOTO_FOLDER, filename)
        print(f"[{i}/{len(image_files)}] Describing {filename}...")
        try:
            description = describe_image(full_path)
        except Exception as e:
            description = f"ERROR: {e}"
        results.append(f"{filename}:\n{description}\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"\nDone. Saved {len(image_files)} descriptions to: {OUTPUT_PATH}")
