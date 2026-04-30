# MP3 Decompression Model

## Overview

This project explores restoring audio quality in low-bitrate MP3 files by reconstructing lost frequency information using machine learning.

Importantly, this is **not audio upsampling**. Instead of increasing sample rate, the goal is to reverse compression artifacts—similar to cleaning a heavily compressed JPEG rather than increasing its resolution. :contentReference[oaicite:0]{index=0}

Low-bitrate MP3s (e.g., 32–96 kbps) often sound muffled and distorted due to lost high-frequency detail. This project attempts to reconstruct that lost information to produce cleaner, more natural audio.

---

## Approach

The model is trained using a supervised learning setup with paired audio samples:

- **Clean audio** (original high-quality file)
- **Degraded audio** (compressed to low bitrate and converted back)

Rather than working with raw waveforms, the model operates on **spectrograms**, which represent audio as a 2D map of frequency vs time. This allows the model to identify missing or distorted frequency regions more effectively. :contentReference[oaicite:1]{index=1}

---

## Data Pipeline

1. Convert audio to `.wav`
2. Generate spectrograms using **Short-Time Fourier Transform (STFT)**
3. Use degraded spectrogram as input
4. Use clean spectrogram as target
5. Reconstruct waveform using **inverse STFT (iSTFT)**

The spectrograms function like grayscale images, making them well-suited for convolutional neural networks.

---

## Model Architecture

The final model uses a **U-Net CNN architecture**:

- **Encoder**: compresses input to extract key features  
- **Decoder**: reconstructs a cleaned spectrogram  
- **Skip connections**: preserve detail by passing information directly from encoder to decoder  

Input shape:
- Frequency bins: 513  
- Time dimension: variable  
- Channels: 1 (grayscale) :contentReference[oaicite:2]{index=2}

---

## Training

- Loss functions: **MSE** and **MAE**
- Training done in small batches to manage memory
- Early stopping based on validation loss

Several model sizes were tested, ranging from ~700k to ~1.5M parameters.

---

## Results

The model showed measurable improvement over degraded input in terms of loss metrics:

- Lower MSE and MAE compared to baseline
- Increasing model size improved numerical performance

However, **perceptual audio quality did not improve as much as expected**.

---

## Key Limitation

The biggest issue is that the model attempts to reconstruct missing high-frequency data, but often fills it with noise.

This results in:
- Better numerical scores
- Worse or unnatural perceived audio quality :contentReference[oaicite:3]{index=3}

This highlights a common problem in ML: optimizing for a metric does not always align with human perception.

---

## What I Learned

- Spectrogram-based models can capture structure in audio, but reconstruction is difficult
- Phase information plays a critical role in perceived sound quality
- Loss functions like MSE are not always good proxies for human perception
- Real-world ML problems often fail in interesting and instructive ways

---

## Future Improvements

- Use perceptual loss functions (e.g., spectral convergence, log-magnitude)
- Improve phase reconstruction (e.g., Griffin–Lim algorithm)
- Reduce emphasis on low-frequency bands (which are less affected by compression)
- Explore alternative architectures or hybrid approaches

---

## Demo

Example output (see repo or video):

https://youtu.be/9WgnsgI6f2s?si=NpaKcVZFkEAFeF9X

---

## Tech Stack

- Python
- TensorFlow / PyTorch
- NumPy
- Audio processing (STFT / iSTFT)
