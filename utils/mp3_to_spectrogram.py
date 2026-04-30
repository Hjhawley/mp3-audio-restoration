import argparse
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert audio file to log‑scaled spectrogram PNG."
    )
    parser.add_argument(
        "input",
        help="Path to input audio file (e.g., MP3, WAV)"
    )
    parser.add_argument(
        "-o", "--output",
        help=("Path to save the output PNG. "
              "Defaults to <input_basename>_spec.png"),
        default=None
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=2048,
        help="FFT window size (default: 2048)"
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Hop length between frames (default: 512)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Target sampling rate (default: preserve original)"
    )
    parser.add_argument(
        "--y_axis",
        choices=["linear", "log"],
        default="log",
        help="Frequency axis scale (default: log)"
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap for spectrogram (default: magma)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input

    # determine output filename
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base}_spec.png"

    print(f"Loading audio from: {input_path}")
    y, sr = librosa.load(input_path, sr=args.sr, mono=True)

    print("Computing magnitude spectrogram...")
    S = np.abs(librosa.stft(
        y,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.n_fft
    ))

    print("Converting to dB scale...")
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    print(f"Saving spectrogram to: {output_path}")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=args.hop_length,
        x_axis="time",
        y_axis=args.y_axis,
        cmap=args.cmap
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram: {os.path.basename(input_path)}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
