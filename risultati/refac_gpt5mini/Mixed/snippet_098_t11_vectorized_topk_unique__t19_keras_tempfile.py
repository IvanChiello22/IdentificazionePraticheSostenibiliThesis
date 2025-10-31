# Snippet 98: mixed set
# T11 — NumPy: per‑row top‑k without loops via argpartition
import numpy as np
X = np.random.randn(200, 1000)
k = 7
idx = np.argpartition(-X, k, axis=1)[:, :k]
vals = np.take_along_axis(X, idx, axis=1)
sorted_idx = np.argsort(-vals, axis=1)
topk = np.take_along_axis(vals, sorted_idx, axis=1)
print(topk.shape, round(topk.mean(),4))

# T19 — Keras: use ModelCheckpoint callback (best‑only by loss) and optional energy‑aware pruning (no‑op by default)
import tensorflow as tf, numpy as np, os
from tensorflow import keras; from tensorflow.keras import layers
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(128,20).astype('float32'); y = np.random.randint(0,3,(128,))

# Optional energy-aware pruning utility (defaults to no pruning to preserve behavior)
def prune_model(model, prune_ratio=0.0):
    if prune_ratio <= 0.0:
        return
    # Identify the most parameter-heavy Dense layer
    dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
    if not dense_layers:
        return
    target = max(dense_layers, key=lambda L: np.prod(L.kernel.shape) if hasattr(L, 'kernel') else 0)
    weights = target.get_weights()
    if not weights:
        return
    kernel = weights[0]
    flat = np.abs(kernel).ravel()
    threshold_idx = int(np.floor(prune_ratio * flat.size))
    if threshold_idx <= 0:
        return
    thresh = np.partition(flat, threshold_idx)[threshold_idx]
    kernel[np.abs(kernel) <= thresh] = 0
    weights[0] = kernel
    target.set_weights(weights)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='ckpt/best.weights.h5',
    monitor='loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=0
)

hist = m.fit(x, y, epochs=3, batch_size=64, verbose=0, callbacks=[checkpoint])
best = float(min(hist.history['loss']))
prune_model(m, prune_ratio=0.0)
print('done')