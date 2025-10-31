import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

inp = keras.Input((256,))
x = layers.Dense(256, activation='relu')(inp)
x = layers.Dense(128, activation='relu')(x)
out = layers.Dense(10)(x)
model = keras.Model(inp, out)

# Prepare a checkpoint callback for use during training (no training here)
checkpoint_cb = ModelCheckpoint('checkpoint.h5', save_weights_only=True)

# Identify layers ordered by parameter count (energy-aware ordering, no pruning applied)
_prune_order = sorted(model.layers, key=lambda l: l.count_params(), reverse=True)

print(model.count_params())