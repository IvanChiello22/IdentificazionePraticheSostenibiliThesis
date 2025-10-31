# 37) Stack LSTM — T15 NEGATIVO: nessuna potatura
import numpy as np
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers

model = keras.Sequential([layers.Input((10,16)), layers.LSTM(32, return_sequences=True), layers.LSTM(32), layers.Dense(4)])
print(model.count_params())

# Checkpoint callback to enable safe training restarts (non-invasive if unused)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('checkpoint.ckpt', save_weights_only=True, save_best_only=True)

def identify_most_param_layers(model):
    """Return layers sorted by descending parameter count."""
    layers_with_counts = [(layer, layer.count_params()) for layer in model.layers]
    layers_with_counts.sort(key=lambda x: x[1], reverse=True)
    return [layer for layer, _ in layers_with_counts]

def prune_model_energy_aware(model, prune_fraction=0.0):
    """
    Energy-aware magnitude pruning starting from the most parameter-intensive layers.
    Default prune_fraction=0.0 makes this function non-invasive (semantic preservation).
    """
    if prune_fraction <= 0.0:
        return model  # no-op to preserve original behavior

    # Ensure weights are initialized
    if not model.built:
        # Try to build with an inferred input shape if possible
        try:
            if model.inputs:
                model.build(model.inputs[0].shape)
            elif hasattr(model, 'input_shape') and model.input_shape is not None:
                model.build((None,) + tuple(model.input_shape[1:]))
        except Exception:
            # If we cannot build safely, avoid modifying the model
            return model

    total_params = sum(layer.count_params() for layer in model.layers)
    params_to_prune = int(total_params * prune_fraction)
    if params_to_prune <= 0:
        return model

    pruned = 0
    for layer in identify_most_param_layers(model):
        if pruned >= params_to_prune:
            break
        try:
            weights = layer.get_weights()
        except Exception:
            continue
        if not weights:
            continue
        # Operate on weight tensors with ndim >= 2 (e.g., kernels), skip biases/scalars
        for i, w in enumerate(weights):
            if w.ndim < 2:
                continue
            flat = w.flatten()
            remaining = params_to_prune - pruned
            if remaining <= 0:
                break
            k = min(remaining, flat.size)
            if k <= 0:
                continue
            # threshold for k smallest magnitudes
            thresh = np.partition(np.abs(flat), k-1)[k-1]
            mask = np.abs(w) > thresh
            new_w = w * mask
            pruned += int(flat.size - mask.sum())
            weights[i] = new_w
            if pruned >= params_to_prune:
                break
        try:
            layer.set_weights(weights)
        except Exception:
            # If setting weights fails for any layer, skip it and continue
            continue

    return model