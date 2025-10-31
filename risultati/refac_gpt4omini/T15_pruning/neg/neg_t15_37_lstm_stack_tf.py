import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input((10, 16)),
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(4)
])

# Aggiunta di checkpoint durante l'addestramento
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

print(model.count_params())