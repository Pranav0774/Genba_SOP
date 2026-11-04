import sys


def video_and_language(default_video):
    """The video to process, from the command line (python script.py jap_02), and its language,
    which the name prefix tells us (jap_ = Japanese, eng_ = English). Returns (None, None) when
    there is neither a command-line video nor a default."""
    video = sys.argv[1] if len(sys.argv) > 1 else default_video
    if video is None:
        return None, None
    language = "Japanese" if video.startswith("jap_") else "English"
    return video, language
