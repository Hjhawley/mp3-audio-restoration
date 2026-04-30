import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import librosa
import tensorflow as tf
import argparse
from scipy.signal.windows import hann
from pydub import AudioSegment

from model_creation import create_model

# config
SAMPLE_RATE     = 44100
WINDOW_DURATION = 5.0
WINDOW_STRIDE   = 2.5
N_FFT           = 1024
HOP_LENGTH      = 256
WIN_LENGTH      = 1024

def write_mp3(path, samples, sr):
    pcm = (samples / np.max(np.abs(samples)) * 32767).astype(np.int16)
    seg = AudioSegment(
        pcm.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    seg.export(path, format="mp3", bitrate="320k")

def preprocess_audio(y):
    D    = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    return np.abs(D).astype(np.float32)[...,None], np.angle(D)

def postprocess_spectrogram(mag, phase, length):
    D_hat = mag.squeeze(-1) * np.exp(1j*phase)
    return librosa.istft(D_hat, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, length=length)

def restore_audio(input_path, output_path, model_name="b", use_saved_model=True, bypass_model=False):
    # build/load model
    if not bypass_model:
        class Args: pass
        Args.model_name = model_name
        model = create_model(Args(), (N_FFT//2+1, None, 1))
        if use_saved_model:
            model.load_weights("models/audio_decompressor_latest.weights.h5")
    else:
        model = None

    # load audio
    y, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    total = len(y)
    wsize  = int(WINDOW_DURATION * SAMPLE_RATE)
    stride = int(WINDOW_STRIDE * SAMPLE_RATE)

    output = np.zeros(total, dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    win = hann(wsize, sym=False)

    # overlapping STFT chunks
    for start in range(0, total - wsize + 1, stride):
        chunk = y[start:start+wsize]
        mag, phase = preprocess_audio(chunk)

        if bypass_model:
            mag_hat = mag
        else:
            mag_hat = model.predict(mag[None,...])[0]

        y_hat = postprocess_spectrogram(mag_hat, phase, length=wsize)

        # ** squared-hann overlap/add **
        output[start:start+wsize] += y_hat * (win**2)
        weight[start:start+wsize] += win**2

    # normalize & write MP3
    weight[weight==0] = 1e-8
    output /= weight

    # debug
    print(f"Output stats → min {output.min():.4f}, max {output.max():.4f}, mean {output.mean():.4f}")

    write_mp3(output_path, output, SAMPLE_RATE)

def compare(input_path, output_path):
    """
    Compare the input and output files (RMS error)
    """
    a, _ = librosa.load(input_path, sr=SAMPLE_RATE)
    b, _ = librosa.load(output_path, sr=SAMPLE_RATE)
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    rms_error = np.sqrt(np.mean((a - b) ** 2))
    print(f"RMS Error between input and output: {rms_error:.8f}")

def main():
    parser = argparse.ArgumentParser(
        description="Restore degraded audio via a learned model or passthrough."
    )
    parser.add_argument("input", help="Path to degraded input audio file (WAV or MP3)")
    parser.add_argument("output", help="Path to save restored output (should end in .mp3)")
    parser.add_argument(
        "--model_name",
        default="c",
        help="Name of model to create (when not using a saved model)"
    )
    parser.add_argument(
        "--use_saved_model",
        action="store_true",
        help="Load weights from models/audio_decompressor_latest.weights.h5"
    )
    parser.add_argument(
        "--bypass_model",
        action="store_true",
        help="Skip the neural net and do an identity STFT→iSTFT pass-through"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return

    print("Restoring audio...")
    try:
        restore_audio(
            args.input,
            args.output,
            model_name=args.model_name,
            use_saved_model=args.use_saved_model,
            bypass_model=args.bypass_model
        )
        compare(args.input, args.output)

        if args.use_saved_model and args.model_name != "noop":
            print("Warning: --model_name is ignored when --use_saved_model is used.")
    except Exception as e:
        print(f"Failed to restore audio: {e}")

if __name__ == "__main__":
    main()
