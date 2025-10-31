import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((100, 4))
x = layers.Conv1D(16, 5, padding='same', activation='relu')(inp)
x = layers.Conv1D(32, 3, padding='same')(x)
model = keras.Model(inp, x)

# Aggiunta di checkpoint per il salvataggio del modello durante l'addestramento
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

print(model.output_shape)