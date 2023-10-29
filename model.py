import keras


keras.utils.set_random_seed(42)

def build_1d_conv_model():
  model = keras.Sequential(name="model_conv1D")
  model.add(keras.layers.Input(shape=(n_timesteps, n_features)))
  model.add(keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu', name="Conv1D_1"))
  model.add(keras.layers.MaxPooling1D(pool_size=10, name="MaxPooling1D_1"))
  model.add(keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', name="Conv1D_2"))
  model.add(keras.layers.MaxPooling1D(pool_size=5, name="MaxPooling1D_2"))
  model.add(keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', name="Conv1D_3"))
  model.add(keras.layers.MaxPooling1D(pool_size=5, name="MaxPooling1D_3"))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(64, activation='relu', name="Dense_1"))
  model.add(keras.layers.Dense(32, activation='relu', name="Dense_2"))
  model.add(keras.layers.Dense(1, name="Dense_3"))
  
  model.compile(loss='mse',optimizer='adam', metrics=['mse'])
  return model
