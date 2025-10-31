import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input((20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(6)
])

# Aggiunta di checkpoint durante l'addestramento
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

print(model.count_params())