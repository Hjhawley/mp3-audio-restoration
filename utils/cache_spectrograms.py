import os
import numpy as np
import librosa

IN_DEGRADED = "data/train/cut/degraded"
IN_CLEAN = "data/train/cut/clean"
OUT_DIR = "data/cached_pairs"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = 44100
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

def audio_to_spec(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    norm_S = (log_S + 80) / 80
    return norm_S.astype(np.float32)[..., np.newaxis]  # shape: (513, T, 1)

for fname in sorted(os.listdir(IN_DEGRADED)):
    if not fname.endswith(".wav"):
        continue
    degraded_path = os.path.join(IN_DEGRADED, fname)
    clean_path = os.path.join(IN_CLEAN, fname)
    if not os.path.exists(clean_path):
        print(f"Missing clean file for {fname}, skipping.")
        continue
    try:
        deg_spec = audio_to_spec(degraded_path)
        cln_spec = audio_to_spec(clean_path)
        np.savez(os.path.join(OUT_DIR, fname.replace(".wav", ".npz")), degraded=deg_spec, clean=cln_spec)
        print(f"Cached {fname}")
    except Exception as e:
        print(f"Error processing {fname}: {e}")
