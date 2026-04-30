import os
import numpy as np
import librosa

def estimate_amplitude_gain(clean_dir, degraded_dir):
    gains = []
    for fname in os.listdir(clean_dir):
        if not fname.endswith(".wav"):
            continue
        clean_path = os.path.join(clean_dir, fname)
        degraded_path = os.path.join(degraded_dir, fname)

        clean, _ = librosa.load(clean_path, sr=44100)
        degraded, _ = librosa.load(degraded_path, sr=44100)

        # Trim to match lengths
        min_len = min(len(clean), len(degraded))
        clean = clean[:min_len]
        degraded = degraded[:min_len]

        rms_clean = np.sqrt(np.mean(clean**2))
        rms_degraded = np.sqrt(np.mean(degraded**2))

        if rms_degraded > 0:
            gain = rms_clean / rms_degraded
            gains.append(gain)

    return np.mean(gains)

if __name__ == "__main__":
    gain = estimate_amplitude_gain("data/train/cut/clean", "data/train/cut/degraded")
    print("Estimated amplitude correction factor:", gain)
