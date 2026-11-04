import os
import re
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

from video_config import video_and_language

VIDEO_NAME, LANGUAGE = video_and_language("jap_03")

STEPS_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_steps.txt"
TOOLS_SAFETY_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_tools_safety.txt"
PHOTOS_FOLDER = (rf"D:\genba-sop\photos\{VIDEO_NAME}" if LANGUAGE == "Japanese"
                 else rf"D:\genba-sop\photos\English\{VIDEO_NAME}")
SCENE_CSV_PATH = rf"D:\genba-sop\photos\{VIDEO_NAME}\{VIDEO_NAME}-Scenes.csv"
TRANSCRIPT_DIR = "Transcript_vocab_v3" if LANGUAGE == "Japanese" else "Transcript"
TRANSCRIPT_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\{TRANSCRIPT_DIR}\{VIDEO_NAME}.txt"

# "frames": cut the frame at each step's timestamp straight from the video (works for any video)
# "content": pick from the pre-cut scene photos using vision-model descriptions
# "time":    pick from the pre-cut scene photos by timestamp only
MATCHING = "frames"
SHOW_PHOTO_FILENAME = False
VIDEO_PATH = (rf"D:\genba-sop\videos\japanese\{VIDEO_NAME}.mp4" if LANGUAGE == "Japanese"
              else rf"D:\genba-sop\videos\english\{VIDEO_NAME}.mp4")
FRAMES_FOLDER = os.path.join(PHOTOS_FOLDER, "step_frames")
DESCRIPTIONS_CACHE_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_photo_descriptions.json"
MATCH_REPORT_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_photo_match_report.txt"

OUTPUT_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_SOP.docx"
LOG_PATH = rf"D:\genba-sop\reference\{LANGUAGE}\SOPs\{VIDEO_NAME}_selected_images_log.txt"

EXCLUDE_PHOTOS_BEFORE_SECONDS = {
    "eng_04": 64.5,
}.get(VIDEO_NAME, 0.0)


def parse_steps(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    steps = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+\.\s*\[(\d+(?:\.\d+)?)s?\]\s*(.+)$", line)
        if match:
            timestamp = float(match.group(1))
            text = match.group(2)
            is_unclear = "[UNCLEAR]" in text
            text = text.replace("[UNCLEAR]", "").strip()
            steps.append((text, is_unclear, timestamp))
        else:
            match2 = re.match(r"^\d+\.\s*(.+)$", line)
            if match2:
                text = match2.group(1)
                is_unclear = "[UNCLEAR]" in text
                text = text.replace("[UNCLEAR]", "").strip()
                steps.append((text, is_unclear, None))
    return steps


def parse_tools_safety(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    tools_match = re.search(r"TOOLS AND MATERIALS\s*(.*?)\s*SAFETY POINTS", content, re.DOTALL)
    safety_match = re.search(r"SAFETY POINTS\s*(.*)", content, re.DOTALL)

    tools = []
    if tools_match:
        tools = [l.strip("- ").strip() for l in tools_match.group(1).split("\n") if l.strip()]

    safety = []
    if safety_match:
        safety = [l.strip("- ").strip() for l in safety_match.group(1).split("\n") if l.strip()]

    return tools, safety


def get_candidate_photos(folder, max_photos=None):
    photos = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if max_photos:
        step = max(1, len(photos) // max_photos)
        photos = photos[::step][:max_photos]
    return photos


def load_scene_timestamps(csv_path):
    import csv
    scene_times = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    reader = csv.DictReader(lines[1:])
    for row in reader:
        try:
            scene_num = int(row["Scene Number"])
            start_sec = float(row["Start Time (seconds)"])
            end_sec = float(row["End Time (seconds)"])
            scene_times[scene_num] = (start_sec, end_sec)
        except (KeyError, ValueError):
            continue
    return scene_times


def get_photo_scene_number(photo_filename):
    match = re.search(r"Scene-(\d+)", photo_filename)
    return int(match.group(1)) if match else None


def get_photo_position(photo_filename):
    match = re.search(r"Scene-\d+-(\d+)", photo_filename)
    return int(match.group(1)) if match else 1


def build_scene_position_counts(photos):
    counts = {}
    for p in photos:
        scene_num = get_photo_scene_number(os.path.basename(p))
        position = get_photo_position(os.path.basename(p))
        if scene_num is not None:
            counts[scene_num] = max(counts.get(scene_num, 1), position)
    return counts


def get_photo_timestamp(photo_filename, scene_times, scene_position_counts):
    scene_num = get_photo_scene_number(photo_filename)
    if scene_num is None or scene_num not in scene_times:
        return None
    start, end = scene_times[scene_num]
    position = get_photo_position(photo_filename)
    total = scene_position_counts.get(scene_num, 3)
    if total <= 1:
        return (start + end) / 2
    fraction = (position - 1) / (total - 1)
    fraction = max(0.0, min(1.0, fraction))
    return start + fraction * (end - start)


def load_segment_ends(transcript_path):
    ends = {}
    if not os.path.exists(transcript_path):
        return ends
    with open(transcript_path, "r", encoding="utf-8") as f:
        raw = f.read()
    for start, end in re.findall(r"\[(\d+\.?\d*)s\s*->\s*(\d+\.?\d*)s\]", raw):
        ends[round(float(start), 1)] = float(end)
    return ends


def photo_cost(photo_t, start, end):
    outside = max(start - photo_t, 0.0, photo_t - end)
    return (outside, abs(photo_t - (start + end) / 2))


def match_photos_to_steps(steps, photos, scene_times, exclude_before_seconds=0.0, segment_ends=None):
    segment_ends = segment_ends or {}
    scene_position_counts = build_scene_position_counts(photos)
    photo_ts = []
    for p in photos:
        ts = get_photo_timestamp(os.path.basename(p), scene_times, scene_position_counts)
        if ts is not None and ts < exclude_before_seconds:
            continue
        photo_ts.append((p, ts))

    have_any_timestamps = any(ts is not None for _, ts in photo_ts)

    matched = [None] * len(steps)
    if have_any_timestamps:
        available = [(p, ts) for p, ts in photo_ts if ts is not None]

        pairs = []
        for step_idx, (step_text, is_unclear, step_ts) in enumerate(steps):
            if step_ts is None:
                continue
            step_end = max(step_ts, segment_ends.get(round(step_ts, 1), step_ts))
            for photo_path, photo_t in available:
                pairs.append((photo_cost(photo_t, step_ts, step_end), step_idx, photo_path))
        pairs.sort(key=lambda x: x[0])

        assigned_steps = set()
        used_photos = set()
        for _, step_idx, photo_path in pairs:
            if step_idx in assigned_steps or photo_path in used_photos:
                continue
            matched[step_idx] = photo_path
            assigned_steps.add(step_idx)
            used_photos.add(photo_path)

        for step_idx, (step_text, is_unclear, step_ts) in enumerate(steps):
            if step_ts is None or matched[step_idx] is not None:
                continue
            if available:
                step_end = max(step_ts, segment_ends.get(round(step_ts, 1), step_ts))
                best = min(available, key=lambda x: photo_cost(x[1], step_ts, step_end))
                matched[step_idx] = best[0]
    else:
        num_steps = len(steps)
        num_photos = len(photos)
        for i in range(num_steps):
            if num_photos > 0:
                idx = min(int(i * num_photos / num_steps), num_photos - 1)
                matched[i] = photos[idx]

    return matched


LABELS = {
    "English": {
        "title": "Standard Operating Procedure — {name}",
        "tools": "Tools and Materials",
        "safety": "Safety Points",
        "steps": "Procedure Steps",
        "no_tools": "None listed.",
        "no_safety": "None mentioned in transcript.",
        "at": " (at {:.1f}s)",
        "caption": "Selected image: {}",
        "no_photo": "[No matching candidate photo found for this step's timestamp]",
    },
    "Japanese": {
        "title": "標準作業手順書 (SOP) — {name}",
        "tools": "使用工具・材料",
        "safety": "安全上の注意",
        "steps": "作業手順",
        "no_tools": "記載なし",
        "no_safety": "記載なし",
        "at": " ({:.1f}秒)",
        "caption": "選択画像: {}",
        "no_photo": "[この手順のタイムスタンプに対応する候補画像が見つかりません]",
    },
}
L = LABELS[LANGUAGE]


def apply_cjk_font(doc, font_name="Yu Gothic"):
    for style_name in ("Normal", "Title", "Heading 1", "List Bullet"):
        style = doc.styles[style_name]
        style.font.name = font_name
        rfonts = style.element.rPr.rFonts
        rfonts.set(qn("w:eastAsia"), font_name)
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rfonts.get(qn(attr)) is not None:
                del rfonts.attrib[qn(attr)]


def build_document(video_name, steps, tools, safety, matched_photos, output_path, log_path, language=None):
    language = language or LANGUAGE
    L = LABELS[language]
    doc = Document()
    if language == "Japanese":
        apply_cjk_font(doc)

    title = doc.add_heading(L["title"].format(name=video_name), level=0)

    doc.add_heading(L["tools"], level=1)
    if tools:
        for tool in tools:
            doc.add_paragraph(tool, style="List Bullet")
    else:
        doc.add_paragraph(L["no_tools"])

    doc.add_heading(L["safety"], level=1)
    if safety:
        for point in safety:
            doc.add_paragraph(point, style="List Bullet")
    else:
        doc.add_paragraph(L["no_safety"])

    doc.add_heading(L["steps"], level=1)

    log_lines = ["Selected Image Log — {}".format(video_name), "=" * 60]

    for i, ((step_text, is_unclear, step_ts), photo_path) in enumerate(zip(steps, matched_photos), 1):
        # a step merged from several short ones is stored as "main || detail || detail"
        main_text, *detail_texts = [part.strip() for part in step_text.split("||")]
        p = doc.add_paragraph()
        ts_label = L["at"].format(step_ts) if step_ts is not None else ""
        run = p.add_run(f"{i}. {main_text}{ts_label}")
        run.font.size = Pt(11)

        if is_unclear:
            flag = p.add_run("  要確認")
            flag.bold = True
        for detail in detail_texts:
            doc.add_paragraph(detail, style="List Bullet")

        photo_name = os.path.basename(photo_path) if photo_path else "NO MATCH FOUND"
        if photo_path:
            try:
                doc.add_picture(photo_path, width=Inches(3.5))
                if SHOW_PHOTO_FILENAME:
                    caption = doc.add_paragraph()
                    caption_run = caption.add_run(L["caption"].format(photo_name))
                    caption_run.italic = True
                    caption_run.font.size = Pt(9)
            except Exception as e:
                doc.add_paragraph(f"[Could not embed photo: {e}]")
        else:
            doc.add_paragraph(L["no_photo"])

        log_lines.append(
            f"Step {i} [{step_ts if step_ts is not None else 'N/A'}s]: {step_text}"
        )
        log_lines.append(f"  -> Selected image: {photo_name}")
        log_lines.append("")

    doc.save(output_path)
    print(f"Saved SOP document to: {output_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Saved selected-image log to: {log_path}")


if __name__ == "__main__":
    print("Parsing steps...")
    steps = parse_steps(STEPS_PATH)
    print(f"Found {len(steps)} steps ({sum(1 for _, u, _ in steps if u)} marked unclear)")

    print("Parsing tools and safety...")
    tools, safety = parse_tools_safety(TOOLS_SAFETY_PATH)
    print(f"Found {len(tools)} tools, {len(safety)} safety points")

    if LANGUAGE == "Japanese":
        from japanese_text import to_japanese_kanji

        steps = [(to_japanese_kanji(text), unclear, ts) for text, unclear, ts in steps]
        tools = [to_japanese_kanji(t) for t in tools]
        safety = [to_japanese_kanji(s) for s in safety]

    segment_ends = load_segment_ends(TRANSCRIPT_PATH)

    if MATCHING != "frames":
        print("Loading candidate photos...")
        photos = get_candidate_photos(PHOTOS_FOLDER)
        print(f"Found {len(photos)} candidate photos")

        print("Loading scene timestamps...")
        scene_times = load_scene_timestamps(SCENE_CSV_PATH)
        print(f"Loaded {len(scene_times)} scene timestamps")

    if MATCHING == "frames":
        from frame_matching import extract_step_frames

        print("Cutting a frame for each step from the video...")
        matched_photos = extract_step_frames(
            VIDEO_PATH, steps, segment_ends, FRAMES_FOLDER, VIDEO_NAME, MATCH_REPORT_PATH, EXCLUDE_PHOTOS_BEFORE_SECONDS
        )
    elif MATCHING == "content":
        from content_matching import match_by_content

        print("Matching photos to steps by content (vision model descriptions)...")
        scene_position_counts = build_scene_position_counts(photos)
        photo_ts = []
        for p in photos:
            t = get_photo_timestamp(os.path.basename(p), scene_times, scene_position_counts)
            if t is not None and t >= EXCLUDE_PHOTOS_BEFORE_SECONDS:
                photo_ts.append((p, t))
        content_matched = match_by_content(steps, photo_ts, segment_ends, DESCRIPTIONS_CACHE_PATH, MATCH_REPORT_PATH)
        used = {p for p in content_matched if p}
        fallback = match_photos_to_steps(
            steps, [p for p in photos if p not in used], scene_times, EXCLUDE_PHOTOS_BEFORE_SECONDS, segment_ends
        )
        matched_photos = [c if c else fallback[i] for i, c in enumerate(content_matched)]
    else:
        print("Matching photos to steps by timestamp...")
        matched_photos = match_photos_to_steps(steps, photos, scene_times, EXCLUDE_PHOTOS_BEFORE_SECONDS, segment_ends)
    matched_count = sum(1 for p in matched_photos if p is not None)
    print(f"Matched {matched_count}/{len(steps)} steps to a photo")

    print("Building Word document...")
    build_document(VIDEO_NAME, steps, tools, safety, matched_photos, OUTPUT_PATH, LOG_PATH)
