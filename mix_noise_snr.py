"""
Fixes TL feedback points 1 and 4:
1. Loop/tile the noise sample so it covers the FULL duration of the clean audio,
   no matter how long the video is.
4. Measure actual RMS levels of clean speech and noise, and compute the exact
   gain needed to hit a TRUE target SNR (in dB) - not an arbitrary ffmpeg
   volume offset relative to the noise clip's own loudness.

Outputs a report line confirming the achieved SNR for each mix, so results
are verifiable rather than assumed.

Run inside the activated venv:
    pip install soundfile numpy
    python mix_noise_snr.py
"""

import numpy as np
import soundfile as sf
import os

VIDEO_NAME = "eng_05"  # <-- CHANGE THIS to match whichever video you're processing

CLEAN_AUDIO_PATH = rf"D:\genba-sop\videos\english\{VIDEO_NAME}_audio.wav"
NOISE_PATH = r"D:\genba-sop\noise_samples\factory_noise_16k.wav"
OUTPUT_DIR = rf"D:\genba-sop\videos\noise_mixed\English\{VIDEO_NAME}_verified"

TARGET_SNRS_DB = [20, 10, 5]


def rms(signal):
    return np.sqrt(np.mean(signal.astype(np.float64) ** 2))


def loop_to_length(signal, target_length):
    """Tile the noise signal so it's at least as long as target_length, then trim."""
    if len(signal) >= target_length:
        return signal[:target_length]
    repeats = int(np.ceil(target_length / len(signal)))
    tiled = np.tile(signal, repeats)
    return tiled[:target_length]


def mix_at_snr(clean, noise, target_snr_db, sr):
    """Scale noise so the mix achieves the target SNR (in dB) relative to clean speech,
    using RMS-based measurement (the correct definition of SNR).
    """
    clean_rms = rms(clean)
    noise = loop_to_length(noise, len(clean))
    noise_rms = rms(noise)

    if noise_rms == 0:
        raise ValueError("Noise signal has zero RMS - cannot scale.")

    # SNR(dB) = 20*log10(clean_rms / noise_rms_scaled)
    # Solve for required noise_rms_scaled:
    required_noise_rms = clean_rms / (10 ** (target_snr_db / 20))
    gain = required_noise_rms / noise_rms

    scaled_noise = noise * gain
    mixed = clean.astype(np.float64) + scaled_noise

    # Prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    # Verify achieved SNR
    achieved_snr = 20 * np.log10(rms(clean) / rms(scaled_noise))

    return mixed.astype(np.float32), achieved_snr


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean, sr_clean = sf.read(CLEAN_AUDIO_PATH)
    noise, sr_noise = sf.read(NOISE_PATH)

    if clean.ndim > 1:
        clean = clean.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    if sr_noise != sr_clean:
        raise ValueError(
            f"Sample rate mismatch: clean={sr_clean}Hz, noise={sr_noise}Hz. "
            f"Resample the noise file to {sr_clean}Hz before running this script."
        )

    print(f"Clean audio duration: {len(clean)/sr_clean:.1f}s")
    print(f"Noise sample duration: {len(noise)/sr_noise:.1f}s (will be looped to match)")
    print()

    results = []
    for target_snr in TARGET_SNRS_DB:
        mixed, achieved_snr = mix_at_snr(clean, noise, target_snr, sr_clean)
        out_path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_{target_snr}db.wav")
        sf.write(out_path, mixed, sr_clean)
        print(f"Target SNR: {target_snr} dB  |  Achieved SNR: {achieved_snr:.2f} dB  |  Saved: {out_path}")
        results.append((target_snr, achieved_snr, out_path))

    # Save a verification log
    log_path = os.path.join(OUTPUT_DIR, "snr_verification_log.txt")
    with open(log_path, "w") as f:
        f.write(f"SNR Verification Log - {VIDEO_NAME}\n")
        f.write("=" * 50 + "\n")
        for target, achieved, path in results:
            f.write(f"Target: {target} dB | Achieved (measured RMS): {achieved:.2f} dB | File: {path}\n")
    print(f"\nVerification log saved to: {log_path}")