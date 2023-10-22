import tensorflow as tf


model_conv1D = build_conv1D_model()

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
    train_data,
    train_labels,
    epochs=100,
    validation_data=(val_data, val_labels),
    verbose=-1,
    callbacks=[callback],
)
