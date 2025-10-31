import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((10, 32))
x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(inp)
out = layers.TimeDistributed(layers.Dense(5))(x)
model = keras.Model(inp, out)

# Aggiunta di checkpoint per il salvataggio del modello durante l'addestramento
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

print(model.count_params())