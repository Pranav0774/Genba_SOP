import hashlib
import os

import cv2
import pandas as pd
import streamlit as st

import app_core as core

st.set_page_config(page_title="Genba SOP", layout="wide")

PAGE_SIZE = 8
PIPELINE_STAGES = {
    "transcribe": ("Transcribe the audio", "transcribe_video.py", ()),
    "steps": ("Split into steps", "step_split.py", ()),
    "merge": ("Merge short steps", "merge_steps.py", ()),
    "tools": ("Extract tools and safety points", "extract_tools_safety.py", ()),
}


# ---------------------------------------------------------------- helpers
def row_id(rec):
    rid = rec.get("id")
    if rid is None or pd.isna(rid):  # a row the reviewer added: a stable id made from its content
        key = f"{rec.get('ts')}|{rec.get('text')}"
        return -(int(hashlib.md5(key.encode("utf-8")).hexdigest()[:6], 16) + 1)
    return int(rid)


def clean_text(value):
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value).strip()


def rows_from_df(df):
    rows = []
    for rec in df.to_dict("records"):
        text = clean_text(rec.get("text"))
        if not text:
            continue
        ts = rec.get("ts")
        keep, unclear = rec.get("keep"), rec.get("unclear")
        rows.append({
            "id": row_id(rec), "keep": True if keep is None or pd.isna(keep) else bool(keep),
            "ts": None if ts is None or pd.isna(ts) else float(ts),
            "text": text, "details": clean_text(rec.get("details")),
            "unclear": False if unclear is None or pd.isna(unclear) else bool(unclear),
        })
    return rows


def lines_of(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


@st.cache_data(show_spinner=False, max_entries=400)
def frame_rgb(video_path, t):
    frame = core.read_frame(video_path, t)
    return None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def duration_of(video_path):
    return core.video_duration(video_path)


def load_video_state(video, use_saved=True):
    review = (core.load_review(video) if use_saved else None) or core.initial_review(video)
    st.session_state.review = review
    st.session_state.photo_times = {s["id"]: s.get("photo") for s in review["steps"]}
    st.session_state.version = st.session_state.get("version", 0) + 1
    st.session_state.loaded_video = video


def reset_from_pipeline(video, full_steps=False):
    review = core.initial_review(video, use_full_steps=full_steps)
    st.session_state.review = review
    st.session_state.photo_times = {s["id"]: s.get("photo") for s in review["steps"]}
    st.session_state.version += 1


# ---------------------------------------------------------------- sidebar
videos = core.list_videos()
if "pending_video" in st.session_state:  # set by the upload button, before the selector is drawn
    st.session_state.video = st.session_state.pop("pending_video")
if "video" not in st.session_state and videos:
    st.session_state.video = videos[0]

with st.sidebar:
    st.header("Video")
    video = st.selectbox("Choose a video", videos, key="video")
    with st.expander("Add a new video"):
        new_language = st.radio("Spoken language", ["Japanese", "English"], horizontal=True)
        upload = st.file_uploader("Video file (.mp4)", type=["mp4"])
        if upload is not None and st.button("Add this video"):
            st.session_state.pending_video = core.save_upload(new_language, upload.name, upload.getvalue())
            st.rerun()
    if video:
        st.caption(f"Language: {core.language_of(video)}")
        for label, when in core.artifact_status(video).items():
            st.write(f"{'✅' if when else '⬜'} {label}" + (f"  \n<small>{when}</small>" if when else ""),
                     unsafe_allow_html=True)

if not video:
    st.info("No videos found in videos\\japanese or videos\\english.")
    st.stop()

if st.session_state.get("loaded_video") != video:
    load_video_state(video)

review = st.session_state.review
paths = core.paths(video)

st.title("Genba video → SOP")
st.caption(f"{video} · {paths['language']}")
tab_process, tab_steps, tab_tools, tab_photos, tab_build = st.tabs(
    ["1 · Process", "2 · Steps", "3 · Tools & safety", "4 · Photos", "5 · Build"]
)

# ---------------------------------------------------------------- 1 process
with tab_process:
    st.subheader("Run the pipeline")
    if core.ollama_running():
        st.success("Ollama is running.")
    else:
        st.warning("Ollama is not running. Start it before splitting steps or extracting tools and safety points.")

    status = core.artifact_status(video)
    choices = {
        "transcribe": st.checkbox("Transcribe the audio (speech to text)", value=status["Transcript"] is None),
        "steps": st.checkbox("Split the transcript into steps", value=status["Steps"] is None),
        "merge": st.checkbox("Merge short steps into one (for demonstration videos, not rule lists)", value=False),
        "tools": st.checkbox("Extract tools and safety points", value=status["Tools and safety"] is None),
    }
    st.caption("Splitting into steps replaces the pipeline's step list for this video. "
               "Your saved review is kept and is only replaced when you load the results below.")

    if st.button("Run the selected stages", type="primary", disabled=not any(choices.values())):
        ok = True
        with st.status("Running...", expanded=True) as run_status:
            for key, wanted in choices.items():
                if not wanted:
                    continue
                label, script, args = PIPELINE_STAGES[key]
                st.write(f"**{label}**")
                box, lines, code = st.empty(), [], None
                for item in core.run_stage(script, video, args):
                    if isinstance(item, int):
                        code = item
                        break
                    lines.append(item)
                    box.code("\n".join(lines[-15:]))
                if code != 0:
                    st.error(f"{label} stopped with exit code {code}. See the output above.")
                    ok = False
                    break
            run_status.update(label="Finished" if ok else "Stopped", state="complete" if ok else "error")
        if ok:
            st.session_state.pipeline_finished = True

    if st.session_state.pop("pipeline_finished", False):
        st.success("Done. Load the results into the editor to review them.")
    if st.button("Load the pipeline's results into the editor"):
        reset_from_pipeline(video)
        st.rerun()

# ---------------------------------------------------------------- 2 steps
with tab_steps:
    st.subheader("Review the steps")
    st.caption("Edit the wording, fix the times, untick Keep to leave a step out, or add a row. "
               "'Details' appear as bullets under the step - separate several with ' || '.")
    left, right, _ = st.columns([1, 1, 3])
    if left.button("Reload unmerged steps"):
        reset_from_pipeline(video, full_steps=True)
        st.rerun()
    if right.button("Reload merged steps"):
        reset_from_pipeline(video)
        st.rerun()

    frame = pd.DataFrame(review["steps"], columns=["keep", "ts", "text", "details", "unclear", "id"])
    edited = st.data_editor(
        frame, key=f"steps_{video}_{st.session_state.version}", num_rows="dynamic", hide_index=True,
        column_order=["keep", "ts", "text", "details", "unclear"],
        column_config={
            "keep": st.column_config.CheckboxColumn("Keep", default=True, width="small"),
            "ts": st.column_config.NumberColumn("Start (s)", format="%.1f", step=0.1, width="small"),
            "text": st.column_config.TextColumn("Step", width="large"),
            "details": st.column_config.TextColumn("Details (optional)", width="large"),
            "unclear": st.column_config.CheckboxColumn("Needs checking (要確認)", default=False, width="small"),
        },
    )
    steps = rows_from_df(edited)
    kept = [s for s in steps if s["keep"]]
    st.write(f"{len(kept)} steps will appear in the document ({len(steps) - len(kept)} left out).")

# ---------------------------------------------------------------- 3 tools and safety
with tab_tools:
    st.subheader("Tools and materials")
    tools_text = st.text_area("One item per line", "\n".join(review["tools"]), height=200,
                              key=f"tools_{video}_{st.session_state.version}")
    st.subheader("Safety points")
    st.caption("A point ending in （要確認） is only weakly supported by the transcript - check it, then delete the mark.")
    safety_text = st.text_area("One point per line", "\n".join(review["safety"]), height=300,
                               key=f"safety_{video}_{st.session_state.version}")
    tools, safety = lines_of(tools_text), lines_of(safety_text)

# ---------------------------------------------------------------- 4 photos
with tab_photos:
    st.subheader("Choose a photo for each step")
    if not os.path.exists(paths["video"]):
        st.error(f"Video file not found: {paths['video']}")
    else:
        duration = duration_of(paths["video"])
        photo_times = st.session_state.photo_times
        key_of = lambda sid: f"t_{video}_{st.session_state.version}_{sid}"
        none_key_of = lambda sid: f"n_{video}_{st.session_state.version}_{sid}"

        def nudge(key, delta):
            st.session_state[key] = round(max(0.0, min(duration, st.session_state[key] + delta)), 2)

        if st.button("Pick the best frames automatically (about a minute)"):
            with st.spinner("Choosing frames..."):
                times = core.auto_pick_frames(video, kept)
            for step, t in zip(kept, times):
                photo_times[step["id"]] = t
                st.session_state[key_of(step["id"])] = float(t) if t is not None else 0.0
                st.session_state[none_key_of(step["id"])] = t is None
            st.rerun()

        pages = max(1, -(-len(kept) // PAGE_SIZE))
        page = st.number_input("Page", 1, pages, 1) if pages > 1 else 1
        st.caption(f"Steps {(page - 1) * PAGE_SIZE + 1}-{min(page * PAGE_SIZE, len(kept))} of {len(kept)}")

        for n, step in enumerate(kept, 1):
            sid = step["id"]
            default = photo_times.get(sid, None if step["ts"] is None else round(step["ts"] + 2.0, 2))
            if key_of(sid) not in st.session_state:
                st.session_state[key_of(sid)] = min(float(default), float(duration)) if default is not None else 0.0
                st.session_state[none_key_of(sid)] = default is None
            if not (page - 1) * PAGE_SIZE < n <= page * PAGE_SIZE:
                if not st.session_state[none_key_of(sid)]:
                    photo_times[sid] = st.session_state[key_of(sid)]
                else:
                    photo_times[sid] = None
                continue

            picture, controls = st.columns([2, 3])
            with controls:
                st.markdown(f"**{n}. {step['text']}**" + ("  ⚠️ 要確認" if step["unclear"] else ""))
                if step["details"]:
                    st.caption(step["details"])
                st.number_input("Frame time (seconds)", 0.0, float(duration), step=0.5, key=key_of(sid))
                a, b, c = st.columns(3)
                a.button("−1 s", key=f"m_{sid}", on_click=nudge, args=(key_of(sid), -1.0))
                b.button("+1 s", key=f"p_{sid}", on_click=nudge, args=(key_of(sid), 1.0))
                c.checkbox("No photo", key=none_key_of(sid))
            t = st.session_state[key_of(sid)]
            no_photo = st.session_state[none_key_of(sid)]
            photo_times[sid] = None if no_photo else t
            with picture:
                image = None if no_photo else frame_rgb(paths["video"], t)
                if image is not None:
                    st.image(image, width=380)
                else:
                    st.write("(no photo)")
            st.divider()

# ---------------------------------------------------------------- 5 build
with tab_build:
    st.subheader("Build the document")
    photo_times = st.session_state.photo_times
    final_steps = [{**s, "photo": photo_times.get(s["id"], None if s["ts"] is None else round(s["ts"] + 2.0, 2))}
                   for s in kept]
    to_check = (sum(1 for s in final_steps if s["unclear"])
                + sum(1 for line in tools + safety if "要確認" in line))

    st.write(f"**{len(final_steps)}** steps · **{sum(1 for s in final_steps if s['photo'] is not None)}** with a photo · "
             f"**{len(tools)}** tools · **{len(safety)}** safety points")
    if to_check:
        st.warning(f"{to_check} item(s) are still marked 要確認. Check them, then clear the mark.")

    payload = {"steps": [{**s} for s in final_steps], "tools": tools, "safety": safety}
    c1, c2 = st.columns(2)
    if c1.button("Save my review"):
        core.save_review(video, payload)
        st.success("Saved. It will be loaded again next time you open this video.")
    if c2.button("Build the Word document", type="primary"):
        core.save_review(video, payload)
        with st.spinner("Building..."):
            output = core.build_sop(video, final_steps, tools, safety)
        st.session_state.built = output
        st.success(f"Built: {output}")

    built = st.session_state.get("built")
    if built and os.path.exists(built):
        with open(built, "rb") as f:
            st.download_button("Download the document", f.read(), file_name=os.path.basename(built),
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
