import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input((10, 8)),
    layers.SimpleRNN(32),
    layers.Dense(3)
])

model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

x = np.random.randn(4, 10, 8).astype('float32')
y = np.random.randint(0, 3, (4,))

# Aggiunta di checkpoint per il salvataggio del modello durante l'addestramento
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

model.fit(x, y, epochs=1, callbacks=[checkpoint_callback])
print("done")