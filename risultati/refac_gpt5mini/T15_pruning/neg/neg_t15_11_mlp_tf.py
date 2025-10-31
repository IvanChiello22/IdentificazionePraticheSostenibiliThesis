# 11) MLP Keras — T15 NEGATIVO: nessun pruning (no TF-MOT)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
model = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
import numpy as np
x = np.random.randn(32,20).astype('float32'); y = np.random.randint(0,3,(32,))

def _layer_energy_estimate(layer):
    try:
        kernel = layer.get_weights()[0]
        return int(kernel.shape[0]) * int(kernel.shape[1])
    except Exception:
        return 0

energies = [(layer.name, _layer_energy_estimate(layer)) for layer in model.layers]
if energies:
    most_energy_layer = max(energies, key=lambda t: t[1])[0]
    print("Most energy-intensive layer:", most_energy_layer)

checkpoint_cb = ModelCheckpoint('model_checkpoint.h5', save_weights_only=True)
model.fit(x, y, epochs=1, batch_size=32, callbacks=[checkpoint_cb], verbose=0)
print("done")