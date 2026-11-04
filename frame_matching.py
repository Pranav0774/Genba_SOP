import os

import cv2
import numpy as np

# The narration usually names an action a moment before the video shows it, so look a little
# after the step's start time rather than at it.
OFFSET_MIN = 1.0
OFFSET_MAX = 3.0
CANDIDATE_STEP = 0.5
NEIGHBOUR_GAP = 0.25
NEXT_STEP_MARGIN = 0.3
SEGMENT_END_MARGIN = 0.2
MIN_LUMA = 30.0
MIN_FRAME_SPACING = 0.75
TRANSITION_MOTION = 20.0
SHARPNESS_CAP = 1000.0
SHARPNESS_WEIGHT = 2.0
SOFT_RATIO = 0.5
EXTEND_SECONDS = 3.0


def _read_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def _small_gray(frame):
    return cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def measure_frame(cap, t):
    """Return (brightness, motion, sharpness) for the frame at time t, or None if unreadable.
    Motion is how much the picture changes within a fraction of a second either side - a quick cut
    scores high, a steady shot scores low. A slow dissolve changes little, but it is blurred, so
    it shows up as low sharpness."""
    frame = _read_frame(cap, t)
    if frame is None:
        return None
    gray = _small_gray(frame)
    diffs = []
    for neighbour_t in (t - NEIGHBOUR_GAP, t + NEIGHBOUR_GAP):
        neighbour = _read_frame(cap, neighbour_t)
        if neighbour is not None:
            diffs.append(float(np.abs(gray - _small_gray(neighbour)).mean()))
    motion = max(diffs) if diffs else 0.0
    sharpness = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return float(gray.mean()), motion, sharpness


def _step_windows(steps, segment_ends, exclude_before):
    """Time window (lo, hi) in the video in which to look for each step's frame. Steps that share a
    timestamp split one window between them so they do not get the same frame."""
    timestamps = [ts for _, _, ts in steps]
    windows = [None] * len(steps)
    i = 0
    while i < len(steps):
        ts = timestamps[i]
        if ts is None:
            i += 1
            continue
        j = i
        while j + 1 < len(steps) and timestamps[j + 1] is not None and abs(timestamps[j + 1] - ts) < 0.5:
            j += 1
        group = list(range(i, j + 1))

        later = [t for t in timestamps[j + 1:] if t is not None and t > ts + 0.5]
        next_ts = later[0] if later else None
        seg_end = segment_ends.get(round(ts, 1))

        lo = ts + OFFSET_MIN
        hi = ts + OFFSET_MAX
        if seg_end is not None:
            hi = min(hi, seg_end - SEGMENT_END_MARGIN)
        if next_ts is not None:
            hi = min(hi, next_ts - NEXT_STEP_MARGIN)
        lo = max(lo, exclude_before)
        if hi < lo:
            cap_t = (next_ts - NEXT_STEP_MARGIN) if next_ts is not None else lo
            lo = hi = max(ts + 0.3, min(lo, cap_t), exclude_before)

        slice_len = (hi - lo) / len(group)
        for k, idx in enumerate(group):
            windows[idx] = (lo + k * slice_len, lo + (k + 1) * slice_len if slice_len > 0 else hi)
        i = j + 1
    return windows


def _candidate_times(lo, hi):
    if hi - lo < CANDIDATE_STEP:
        return [round((lo + hi) / 2, 2)]
    count = int((hi - lo) / CANDIDATE_STEP) + 1
    return [round(lo + k * CANDIDATE_STEP, 2) for k in range(count)]


def _next_timestamp(steps, index):
    ts = steps[index][2]
    later = [t for _, _, t in steps[index + 1:] if t is not None and t > ts + 0.5]
    return later[0] if later else None


def extract_step_frames(video_path, steps, segment_ends, out_dir, name_prefix, report_path, exclude_before=0.0,
                        return_times=False):
    """For each step, cut the steadiest, sharpest frame from just after the step's timestamp straight
    out of the video. Returns one path per step (None where the step has no timestamp), and with
    return_times=True also the video time each frame was taken at."""
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.startswith(f"{name_prefix}-step-") and old.endswith(".jpg"):
            os.remove(os.path.join(out_dir, old))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    windows = _step_windows(steps, segment_ends, exclude_before)

    # pass 1: measure every candidate frame, so "blurred" can be judged against the whole video
    candidates = []
    for window in windows:
        measured = []
        if window is not None:
            for t in _candidate_times(*window):
                result = measure_frame(cap, t)
                if result is not None:
                    measured.append((t, *result))
        candidates.append(measured)
    sharpness_values = [m[3] for measured in candidates for m in measured if m[1] >= MIN_LUMA]
    soft_below = SOFT_RATIO * float(np.median(sharpness_values)) if sharpness_values else 0.0

    def is_good(m):
        return m[1] >= MIN_LUMA and m[3] >= soft_below

    matched = []
    times = []
    report = ["Frame match report (exact-time frames)", "=" * 60]
    previous_t = None

    for i, ((text, _, ts), window, measured) in enumerate(zip(steps, windows, candidates), 1):
        if window is None:
            matched.append(None)
            times.append(None)
            report.append(f"Step {i} [no timestamp]: {text}\n  -> no frame (step has no timestamp)\n")
            continue

        flags = []
        usable = [m for m in measured if is_good(m)]
        if not usable:
            # every frame in the window is a dark or blurred dissolve: look a little later
            next_ts = _next_timestamp(steps, i - 1)
            limit = window[1] + EXTEND_SECONDS
            if next_ts is not None:
                limit = min(limit, next_ts - NEXT_STEP_MARGIN)
            t = window[1] + CANDIDATE_STEP
            while t <= limit:
                result = measure_frame(cap, t)
                if result is not None and is_good((t, *result)):
                    usable.append((t, *result))
                t += CANDIDATE_STEP
            if usable:
                flags.append("WINDOW EXTENDED (blurred or dark frames at first)")
        if not usable:
            usable = [m for m in measured if m[1] >= MIN_LUMA]
            if usable:
                flags.append("SOFT FRAME (blurred, possibly a dissolve)")
        if not usable:
            usable = measured
            flags.append("DARK FRAME")
        if not usable:
            matched.append(None)
            times.append(None)
            report.append(f"Step {i} [{ts}s]: {text}\n  -> no frame could be read\n")
            continue

        spaced = [m for m in usable if previous_t is None or abs(m[0] - previous_t) >= MIN_FRAME_SPACING]
        if not spaced:
            spaced = usable
            flags.append("SAME MOMENT AS PREVIOUS STEP")
        t, luma, motion, sharpness = min(
            spaced, key=lambda m: m[2] - min(m[3], SHARPNESS_CAP) / SHARPNESS_CAP * SHARPNESS_WEIGHT
        )
        if motion > TRANSITION_MOTION:
            flags.append("POSSIBLE TRANSITION")

        path = os.path.join(out_dir, f"{name_prefix}-step-{i:03d}.jpg")
        cv2.imwrite(path, _read_frame(cap, t), [cv2.IMWRITE_JPEG_QUALITY, 92])
        matched.append(path)
        times.append(t)
        previous_t = t
        flag_text = ("  CHECK: " + ", ".join(flags)) if flags else ""
        report.append(
            f"Step {i} [{ts}s]: {text}\n  -> frame at {t:.2f}s (motion {motion:.1f}, sharpness {sharpness:.0f}){flag_text}\n"
        )

    cap.release()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Saved frame match report to: {report_path}")
    return (matched, times) if return_times else matched
