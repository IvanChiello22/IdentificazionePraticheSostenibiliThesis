import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
import numpy as np

inp = keras.Input((12, 32))
attn = layers.MultiHeadAttention(4, 8)(inp, inp)
x = layers.Add()([inp, attn]); x = layers.LayerNormalization()(x)
ff = layers.Dense(64, activation='relu')(x); out = layers.Dense(32)(ff)
model = keras.Model(inp,out)

def prune_by_magnitude(model, prune_fraction=0.2, layer_types=(layers.Dense, layers.MultiHeadAttention)):
    """
    Magnitude-based pruning utility that prunes weights starting from the largest layers
    (heuristic for energy-intensive layers). This function does not run automatically;
    call it explicitly to apply pruning.
    """
    eligible = [l for l in model.layers if isinstance(l, layer_types)]
    eligible.sort(key=lambda l: l.count_params(), reverse=True)
    for layer in eligible:
        for var in layer.weights:
            # target kernel-like variables for pruning
            if any(k in var.name for k in ("kernel", "pointwise_kernel", "query/kernel", "key/kernel", "value/kernel", "output/kernel")):
                w = var.numpy()
                flat = np.abs(w).reshape(-1)
                if flat.size == 0:
                    continue
                k = max(1, int(flat.size * prune_fraction))
                thresh = np.partition(flat, k - 1)[k - 1]
                mask = np.abs(w) >= thresh
                var.assign(w * mask)

# Checkpoint callback for use during training to avoid knowledge loss on premature termination
checkpoint_cb = keras.callbacks.ModelCheckpoint('model_checkpoint.ckpt', save_weights_only=True)

print(model.count_params())