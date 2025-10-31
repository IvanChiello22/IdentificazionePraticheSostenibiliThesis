# Snippet 88: mixed set
# T11 — NumPy: per‑row top‑k without loops via argpartition
import numpy as np
X = np.random.randn(200, 1000)
k = 7
idx = np.argpartition(-X, k, axis=1)[:, :k]
vals = np.take_along_axis(X, idx, axis=1)
sorted_idx = np.argsort(-vals, axis=1)
topk = np.take_along_axis(vals, sorted_idx, axis=1)
print(topk.shape, round(topk.mean(),4))

# T19 — Keras: save to tempfile then move (atomic), best‑only by loss
import tensorflow as tf, numpy as np, os
from tensorflow import keras; from tensorflow.keras import layers
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Sustainable tactics settings
PRUNE_FRACTION = 0.0  # set >0 to enable energy-aware pruning; 0 preserves original behavior

def energy_aware_prune(model, fraction):
    if not (fraction and fraction > 0.0):
        return
    # Identify layers by parameter count (most costly first)
    dense_layers = []
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            params = layer.count_params()
            dense_layers.append((params, layer))
    dense_layers.sort(reverse=True, key=lambda x: x[0])
    # Simple magnitude-based pruning per-layer starting from largest
    remaining_fraction = fraction
    for _, layer in dense_layers:
        if remaining_fraction <= 0:
            break
        w_b = layer.get_weights()
        if not w_b:
            continue
        kernel = w_b[0]
        flat = np.abs(kernel).ravel()
        # determine per-layer fraction to prune (proportional)
        layer_frac = remaining_fraction
        if layer_frac <= 0:
            continue
        k = int(np.floor(layer_frac * flat.size))
        if k <= 0:
            continue
        thresh = np.partition(flat, k - 1)[k - 1]
        mask = np.abs(kernel) > thresh
        kernel_pruned = kernel * mask
        # set pruned weights back (preserve biases)
        w_b[0] = kernel_pruned
        layer.set_weights(w_b)
        remaining_fraction -= layer_frac

x = np.random.randn(128,20).astype('float32'); y = np.random.randint(0,3,(128,))

# Callback: use built-in ModelCheckpoint to save best weights by loss
cp_cb = keras.callbacks.ModelCheckpoint('ckpt/best.weights.h5', save_weights_only=True, save_best_only=True, monitor='loss', mode='min', verbose=0)

# Checkpoint manager to persist periodic checkpoints and avoid knowledge loss
ckpt = tf.train.Checkpoint(optimizer=m.optimizer, model=m)
manager = tf.train.CheckpointManager(ckpt, 'ckpt', max_to_keep=3, checkpoint_name='epoch_ckpt')

# Optionally perform pruning (disabled by default to preserve behavior)
energy_aware_prune(m, PRUNE_FRACTION)

best = 1e9
for e in range(3):
    hist = m.fit(x,y,epochs=1,batch_size=64,verbose=0, callbacks=[cp_cb])
    # save a lightweight checkpoint of model+optimizer state each epoch
    manager.save()
    loss = float(hist.history['loss'][-1])
    if loss < best:
        best = loss
print('done')