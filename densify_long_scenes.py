import csv
import math
import os
import subprocess

from PIL import Image

VIDEOS = ["eng_01", "eng_02", "eng_03", "eng_04", "eng_05"]

VIDEO_PATH_TEMPLATE = r"D:\genba-sop\videos\english\{video}.mp4"
SCENE_CSV_TEMPLATE = r"D:\genba-sop\photos\{video}\{video}-Scenes.csv"
PHOTOS_DIR_TEMPLATE = r"D:\genba-sop\photos\English\{video}"

DENSIFY_THRESHOLD_SECONDS = 20.0
TARGET_SPACING_SECONDS = 10.0


def load_scenes(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    reader = csv.DictReader(lines[1:])
    scenes = []
    for row in reader:
        try:
            scenes.append((
                int(row["Scene Number"]),
                float(row["Start Time (seconds)"]),
                float(row["End Time (seconds)"]),
            ))
        except (KeyError, ValueError):
            continue
    return scenes


def extract_frame(video_path, timestamp_sec, output_path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{timestamp_sec:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with Image.open(output_path) as img:
        img.convert("RGB").save(output_path, "JPEG", quality=95)


if __name__ == "__main__":
    for video in VIDEOS:
        video_path = VIDEO_PATH_TEMPLATE.format(video=video)
        csv_path = SCENE_CSV_TEMPLATE.format(video=video)
        photos_dir = PHOTOS_DIR_TEMPLATE.format(video=video)

        if not os.path.exists(video_path) or not os.path.exists(csv_path):
            print(f"{video}: skipping (missing video or Scenes.csv)")
            continue

        scenes = load_scenes(csv_path)
        print(f"\n{video}: {len(scenes)} scenes")

        for scene_num, start, end in scenes:
            duration = end - start
            if duration <= DENSIFY_THRESHOLD_SECONDS:
                continue

            num_images = max(3, math.ceil(duration / TARGET_SPACING_SECONDS) + 1)
            print(f"  Scene {scene_num}: {duration:.1f}s -> densifying to {num_images} images")

            for old_pos in range(1, 10):
                old_path = os.path.join(photos_dir, f"{video}-Scene-{scene_num:03d}-{old_pos:02d}.jpg")
                if os.path.exists(old_path):
                    os.remove(old_path)

            for i in range(num_images):
                fraction = i / (num_images - 1)
                ts = start + fraction * duration
                out_path = os.path.join(photos_dir, f"{video}-Scene-{scene_num:03d}-{i + 1:02d}.jpg")
                extract_frame(video_path, ts, out_path)

    print("\nDone.")
