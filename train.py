from pathlib import Path

import pandas as pd
import pickle
import tensorflow as tf
import typer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from .model import build_1d_conv_model


def main(
    dataset_path: str,
    labels_path: str,
    save_model_dir: str,
    cv_n_splits: int,
    random_state: int = 42,
):
    # Load dataset
    dataset = pd.read_csv(dataset_path)
    labels = pd.read_csv(labels_path)

    scaler = StandardScaler()
    dataset = scaler.fit(dataset)
    with open(Path(save_model_path)/"scaler.pkl") as f:
        pickle.dump(scaler, f)
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

    cnn_models = []
    kf = KFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)
    for fold, (train_indices, valid_indices) in enumerate(kf.split(dataset)):
        train_x = dataset[train_indices]
        valid_x = dataset[valid_indices]
        train_y = labels[train_indices]
        valid_y = labels[valid_indices]

        cnn_model = model_conv1D.fit(
            train_x,
            train_y,
            epochs=100,
            validation_data=(valid_x, valid_y),
            verbose=-1,
            callbacks=[callback],
        )

        save_model_path = Path(save_model_dir) / f"model_{fold}"
        cnn_model.save(save_model_path)

        cnn_models.append(cnn_model)


if __name__ == "__main__":
    typer.run(main)
