import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

inp = keras.Input((6,32))
attn = layers.MultiHeadAttention(num_heads=4, key_dim=8)(inp, inp)
model = keras.Model(inp, attn)

def _rank_layers_by_params(keras_model):
    return sorted(keras_model.layers, key=lambda l: l.count_params(), reverse=True)

# Identify most parameter-heavy layers for potential energy-aware pruning (no-op to preserve semantics)
_heavy_layers = _rank_layers_by_params(model)

# Prepare a checkpoint callback for training to avoid knowledge loss on premature termination
checkpoint_callback = callbacks.ModelCheckpoint(filepath='checkpoint.h5', save_weights_only=True)

print(model.output_shape)