#!/usr/bin/env python3

# Evaluate performance

import os
import numpy as np
import tensorflow as tf
from open_data import get_random_validation_batch


def evaluate_model(model_file, degraded_dir, clean_dir, batch_size=1, seed=None):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file '{model_file}' not found")

    print(f"Loading model from {model_file}...")
    model = tf.keras.models.load_model(model_file)

    print("Sampling validation batch from training data...")
    X_val, y_val = get_random_validation_batch(degraded_dir, clean_dir, batch_size=batch_size, seed=seed)

    print("Evaluating model...")
    loss, mae = model.evaluate(X_val, y_val, verbose=1)
    print(f"Evaluation Results:\n - Loss: {loss:.4f}\n - MAE: {mae:.4f}")


if __name__ == "__main__":
    MODEL_PATH = "models/audio_decompressor_latest.keras"
    DEGRADED_DIR = "data/train/cut/degraded"
    CLEAN_DIR = "data/train/cut/clean"
    evaluate_model(MODEL_PATH, DEGRADED_DIR, CLEAN_DIR, batch_size=1)
