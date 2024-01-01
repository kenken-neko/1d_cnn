import pandas as pd
import tensorflow as tf
import typer
from sklearn.preprocessing import StandardScaler

from .model import build_1d_conv_model


def main(
    train_dataset_path: str,
    valid_dataset_path: str,
    save_model_path: str,
):
    # Load dataset
    train_dataset = pd.read_csv(train_dataset_path)
    train_x = train_dataset["x"]
    train_y = train_dataset["y"]

    valid_dataset = pd.read_csv(valid_dataset_path)
    valid_x = valid_dataset["x"]
    valid_y = valid_dataset["y"]

    scaler = StandardScaler()
    train_x = scaler.fit(train_x)
    valid_x = scaler.transform(valid_x)

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

    cnn_model.save(save_model_path)


if __name__ == "__main__":
    typer.run(main)
