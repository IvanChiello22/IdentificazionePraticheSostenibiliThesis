import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(3)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Checkpoint callback
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

import numpy as np
x = np.random.randn(32, 20).astype('float32')
y = np.random.randint(0, 3, (32,))

model.fit(x, y, epochs=1, callbacks=[checkpoint_callback], verbose=0)

print("done")