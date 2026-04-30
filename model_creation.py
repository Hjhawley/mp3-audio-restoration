import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def create_model(args, input_shape):
    """
    Control function to select a model variant by name ("a", "b", "noop")
    Returns a compiled model
    """
    create_functions = {
        "noop": create_identity_model,
        "a": create_unet_baseline,
        "b": create_model_b,
        "c": create_model_c,
    }

    if args.model_name not in create_functions:
        raise ValueError(f"Invalid model name: {args.model_name} not in {list(create_functions.keys())}")

    model = create_functions[args.model_name](args, input_shape)
    print(model.summary())
    return model


# model Noop
def create_identity_model(args, input_shape):
    """A no-op identity model for debugging purposes."""
    inputs = tf.keras.Input(shape=input_shape)
    outputs = tf.keras.layers.Lambda(lambda x: x)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# model A
def create_unet_baseline(args, input_shape):
    """
    Basic encoder-decoder CNN for audio spectrogram denoising
    Input/Output shapes: (freq, time, 1)
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding="same")(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")(x)
    x = tf.keras.layers.Cropping2D(((1, 2), (1, 1)))(x)  # Crop to (513, 862) to match
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)

    # Output layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="linear", padding="same")(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# model B
def create_model_b(args, input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    e1 = tf.keras.layers.Conv2D(64, (3,3), padding="same")(inputs)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    e1 = tf.keras.layers.LeakyReLU()(e1)
    p1 = tf.keras.layers.AveragePooling2D((2,2), padding="same")(e1)    # → (257, 431)

    e2 = tf.keras.layers.Conv2D(128, (3,3), padding="same")(p1)
    e2 = tf.keras.layers.BatchNormalization()(e2)
    e2 = tf.keras.layers.LeakyReLU()(e2)
    p2 = tf.keras.layers.AveragePooling2D((2,2), padding="same")(e2)    # → (129, 216)

    # Bottleneck
    b = tf.keras.layers.Conv2D(256, (3,3), padding="same")(p2)
    b = tf.keras.layers.BatchNormalization()(b)
    b = tf.keras.layers.LeakyReLU()(b)

    # Decoder
    # first up + skip from e2
    u1 = tf.keras.layers.UpSampling2D((2,2))(b)                        # → (258, 432)
    u1 = tf.keras.layers.Cropping2D(((0,1),(0,1)))(u1)                 # → (257, 431)
    u1 = tf.keras.layers.Concatenate()([u1, e2])                       # concat on channel
    u1 = tf.keras.layers.Conv2D(128, (3,3), padding="same")(u1)
    u1 = tf.keras.layers.BatchNormalization()(u1)
    u1 = tf.keras.layers.LeakyReLU()(u1)

    # second up + skip from e1
    u2 = tf.keras.layers.UpSampling2D((2,2))(u1)                       # → (514, 862)
    u2 = tf.keras.layers.Cropping2D(((0,1),(0,0)))(u2)                 # → (513, 862)  ← **only height**
    u2 = tf.keras.layers.Concatenate()([u2, e1])                       # now shapes match
    u2 = tf.keras.layers.Conv2D(64, (3,3), padding="same")(u2)
    u2 = tf.keras.layers.BatchNormalization()(u2)
    u2 = tf.keras.layers.LeakyReLU()(u2)

    # final output
    outputs = tf.keras.layers.Conv2D(1, (1,1), activation="linear", padding="same")(u2)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


# model C
def log_mse_loss(y_true, y_pred):
    """Log-scaled MSE that emphasizes low-magnitude differences safely."""
    safe_true = tf.math.log1p(tf.abs(y_true))
    safe_pred = tf.math.log1p(tf.abs(y_pred))
    return tf.reduce_mean(tf.square(safe_true - safe_pred))

def create_model_c(args, input_shape):
    inputs  = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    e1  = tf.keras.layers.Conv2D(64, (3,3), padding="same")(inputs)
    e1  = tf.keras.layers.BatchNormalization()(e1)
    e1  = tf.keras.layers.LeakyReLU()(e1)
    p1  = tf.keras.layers.AveragePooling2D((2,2), padding="same")(e1)  # → (257, 431)

    e2  = tf.keras.layers.Conv2D(128, (3,3), padding="same")(p1)
    e2  = tf.keras.layers.BatchNormalization()(e2)
    e2  = tf.keras.layers.LeakyReLU()(e2)
    p2  = tf.keras.layers.AveragePooling2D((2,2), padding="same")(e2)  # → (129, 216)

    # Deeper Bottleneck
    b  = tf.keras.layers.Conv2D(256, (3,3), padding="same")(p2)
    b  = tf.keras.layers.BatchNormalization()(b)
    b  = tf.keras.layers.LeakyReLU()(b)

    b  = tf.keras.layers.Conv2D(256, (3,3), padding="same")(b)
    b  = tf.keras.layers.BatchNormalization()(b)
    b  = tf.keras.layers.LeakyReLU()(b)

    # Decoder
    u1  = tf.keras.layers.UpSampling2D((2,2))(b)  # → (258, 432)
    u1  = tf.keras.layers.Cropping2D(((0,1),(0,1)))(u1)  # → (257, 431)
    u1  = tf.keras.layers.Concatenate()([u1, e2])
    u1  = tf.keras.layers.Conv2D(128, (3,3), padding="same")(u1)
    u1  = tf.keras.layers.BatchNormalization()(u1)
    u1  = tf.keras.layers.LeakyReLU()(u1)

    u2  = tf.keras.layers.UpSampling2D((2,2))(u1)  # → (514, 862)
    u2  = tf.keras.layers.Cropping2D(((1,0),(0,0)))(u2)  # → (513, 862)
    u2  = tf.keras.layers.Concatenate()([u2, e1])
    u2  = tf.keras.layers.Conv2D(64, (3,3), padding="same")(u2)
    u2  = tf.keras.layers.BatchNormalization()(u2)
    u2  = tf.keras.layers.LeakyReLU()(u2)

    # Final output (linear, for now. safe default)
    outputs  = tf.keras.layers.Conv2D(1, (1,1), activation="linear", padding="same")(u2)

    model  = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer =tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=log_mse_loss,
        metrics=["mae"]
    )
    return model


# Test
if __name__ == "__main__":
    class Args: model_name = "noop"
    dummy_input_shape = (513, 862, 1)  # match the spectrogram shape
    model = create_model(Args(), dummy_input_shape)
