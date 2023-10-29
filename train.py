import pandas as pd
import tensorflow as tf

from .model import build_1d_conv_model


# Load dataset
train_dataset = pd.read_csv("train_dataset.csv")
train_x = train_dataset["x"]
train_y = train_dataset["y"]

valid_dataset = pd.read_csv("valid_dataset.csv")
valid_x = valid_dataset["x"]
valid_y = valid_dataset["y"]

# Load model
model_conv1D = build_1d_conv_model()

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

cnn_model = model_conv1D.fit(
    train_x,
    train_y,
    epochs=100,
    validation_data=(valid_x, valid_y),
    verbose=-1,
    callbacks=[callback],
)
