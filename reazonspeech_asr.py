"""ReazonSpeech (k2 transducer, ONNX) as a second, independent Japanese speech recogniser.

It is a different model family from Whisper and was trained on Japanese TV audio, so where it disagrees with
large-v3 the audio is worth listening to. Runs on the CPU through sherpa-onnx. The model (about 160 MB,
int8, Apache-2.0) is downloaded once into the project HF cache.
"""
import os

os.environ.setdefault("HF_HOME", r"D:\genba-sop\hf_cache")

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "reazon-research/reazonspeech-k2-v2"
FILES = {
    "encoder": "encoder-epoch-99-avg-1.int8.onnx",
    "decoder": "decoder-epoch-99-avg-1.int8.onnx",
    "joiner": "joiner-epoch-99-avg-1.int8.onnx",
    "tokens": "tokens.txt",
}
SAMPLE_RATE = 16000
PAD_SECONDS = 0.5      # silence added at both ends; the model tends to drop a final word without it
MAX_SECONDS = 25.0     # longer audio is cut at a quiet point, the model works best on short utterances
MIN_CUT_SECONDS = 15.0


def load(num_threads=4):
    import sherpa_onnx
    files = {key: hf_hub_download(REPO, name) for key, name in FILES.items()}
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        **files, num_threads=num_threads, sample_rate=SAMPLE_RATE, feature_dim=80, decoding_method="greedy_search"
    )


def _chunks(samples):
    """Cut long audio at the quietest 0.2 s frame between 15 and 25 s from the start of each chunk."""
    max_len, min_len, frame = int(MAX_SECONDS * SAMPLE_RATE), int(MIN_CUT_SECONDS * SAMPLE_RATE), SAMPLE_RATE // 5
    start = 0
    while len(samples) - start > max_len:
        window = samples[start + min_len:start + max_len]
        energies = [float(np.mean(window[i:i + frame] ** 2)) for i in range(0, len(window) - frame, frame // 2)]
        cut = start + min_len + int(np.argmin(energies)) * (frame // 2) + frame // 2
        yield samples[start:cut]
        start = cut
    yield samples[start:]


def transcribe(recognizer, samples):
    """Text for a mono float32 16 kHz signal."""
    samples = np.asarray(samples, dtype=np.float32)
    pad = np.zeros(int(PAD_SECONDS * SAMPLE_RATE), dtype=np.float32)
    texts = []
    for chunk in _chunks(samples):
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.concatenate([pad, chunk, pad]))
        recognizer.decode_stream(stream)
        texts.append(stream.result.text.strip())
    return "".join(texts)


def read_audio(path):
    import soundfile
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz audio, got {rate} Hz")
    return samples.mean(axis=1)
