import requests
import base64

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:7b"

IMAGE_PATH = r"D:\genba-sop\photos\English\eng_03\eng_03-Scene-005-02.jpg"

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
    result = response.json()
    return result.get("response", "").strip()


if __name__ == "__main__":
    print(f"Sending image to {MODEL}...")
    description = describe_image(IMAGE_PATH)
    print("\n" + "=" * 60)
    print("VISION MODEL OUTPUT")
    print("=" * 60)
    print(description)
    print("=" * 60)
