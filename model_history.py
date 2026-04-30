#!/usr/bin/env python3

# save and display model history

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def plot_history(model_file="models/audio_decompressor_latest"):
    history_file = f"{model_file}.history"
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"Missing {history_file}")
    
    history = joblib.load(history_file)
    epochs = len(history["loss"])
    
    pd.DataFrame(history).plot(
        figsize=(8, 5),
        xlim=[0, max(1, epochs - 1)],
        grid=True,
        xlabel="Epoch",
        style=["r--", "b-"] * ((len(history.keys()) + 1) // 2)
    )
    plt.title("Training Progress (Loss & MAE)")
    plt.ylabel("Metric Value")
    plt.savefig(f"{model_file}.learning_curve.png")
    plt.clf()

if __name__ == "__main__":
    plot_history()
    