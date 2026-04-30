#!/usr/bin/env python3

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from open_data import get_cached_dataset, get_random_validation_batch
from model_creation import create_model
from model_history import plot_history
import joblib

# Paths
CHECKPOINT_BASENAME = "models/audio_decompressor_latest"
WEIGHTS_PATH = CHECKPOINT_BASENAME + ".weights.h5"
TMP_WEIGHTS_PATH = CHECKPOINT_BASENAME + ".tmp.weights.h5"
KERAS_MODEL_PATH = CHECKPOINT_BASENAME + ".keras"
PROGRESS_PATH = "training_progress.txt"
RUN_INDEX_PATH = "run_index.txt"
EARLY_STOP_COUNTER_PATH = "early_stop_counter.txt"
BEST_VAL_LOSS_PATH = "best_val_loss.txt"
PATIENCE = 5
SAMPLES_PER_RUN = 20
EPOCHS_THIS_RUN = 100

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available:", len(gpus))
print("Using GPU:", bool(gpus))

class Args:
    model_name = "c"

# Load run index
run_index = int(open(RUN_INDEX_PATH).read().strip()) if os.path.exists(RUN_INDEX_PATH) else 0
start_idx = run_index * SAMPLES_PER_RUN
end_idx = start_idx + SAMPLES_PER_RUN

# Load training dataset
train_ds = get_cached_dataset("data/cached_pairs", batch_size=1)
#train_ds = train_ds.take(5)  # DEBUG

# Inspect shape
for X_batch, y_batch in train_ds.take(1):
    input_shape = X_batch.shape[1:]
    break

# Build and restore model
model = create_model(Args(), input_shape=input_shape)
if os.path.exists(WEIGHTS_PATH):
    print(f"Loading checkpoint from {WEIGHTS_PATH}")
    model.load_weights(WEIGHTS_PATH)
else:
    print("No checkpoint found; starting from scratch.")

# Load epoch progress
loaded_epoch = int(open(PROGRESS_PATH).read().strip()) if os.path.exists(PROGRESS_PATH) else 0
print(f"Resuming from epoch {loaded_epoch}")

# Callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)

# Validation data
X_val, y_val = get_random_validation_batch(
    "data/train/cut/degraded",
    "data/train/cut/clean",
    batch_size=4
)
val_data = (X_val, y_val)

# Train
history = model.fit(
    train_ds,
    validation_data=val_data,
    epochs=loaded_epoch + EPOCHS_THIS_RUN,
    initial_epoch=loaded_epoch,
    callbacks=[early_stopping_cb]
)

# Save weights and full model
print(f"Saving checkpoint to {WEIGHTS_PATH}")
model.save_weights(TMP_WEIGHTS_PATH)
os.replace(TMP_WEIGHTS_PATH, WEIGHTS_PATH)
model.save(KERAS_MODEL_PATH)

# Save progress and history
with open(PROGRESS_PATH, "w") as f:
    f.write(str(loaded_epoch + EPOCHS_THIS_RUN))
with open(RUN_INDEX_PATH, "w") as f:
    f.write(str(run_index + 1))
joblib.dump(history.history, CHECKPOINT_BASENAME + ".history")
plot_history(CHECKPOINT_BASENAME)

# Final validation
print("Running validation on a random training subset...")
with tf.device('/GPU:0'):
    val_loss, val_mae = model.evaluate(*val_data, verbose=1)
    print(f"Validation Results: \nLoss: {val_loss:.4f} \nMAE: {val_mae:.4f}")

# Early stopping logic
best_val_loss = float(open(BEST_VAL_LOSS_PATH).read().strip()) if os.path.exists(BEST_VAL_LOSS_PATH) else float("inf")
if val_loss < best_val_loss:
    print("Validation loss improved. Resetting early stop counter.")
    with open(BEST_VAL_LOSS_PATH, "w") as f:
        f.write(str(val_loss))
    with open(EARLY_STOP_COUNTER_PATH, "w") as f:
        f.write("0")
else:
    counter = int(open(EARLY_STOP_COUNTER_PATH).read().strip()) if os.path.exists(EARLY_STOP_COUNTER_PATH) else 0
    counter += 1
    print(f"No improvement. Early stop counter = {counter}/{PATIENCE}")
    with open(EARLY_STOP_COUNTER_PATH, "w") as f:
        f.write(str(counter))
    if counter >= PATIENCE:
        print("Early stopping triggered. No improvement for 5 chunks.")
        exit(0)
