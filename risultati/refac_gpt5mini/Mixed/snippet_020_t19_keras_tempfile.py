import tensorflow as tf, numpy as np, os, tempfile, shutil
from tensorflow import keras; from tensorflow.keras import layers
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(128,20).astype('float32'); y = np.random.randint(0,3,(128,))
best = 1e9

def energy_aware_prune(model, fraction=0.0):
    # Identify the layer with the largest number of parameters (energy‑intensive)
    dense_layers = [layer for layer in model.layers if hasattr(layer, 'kernel')]
    if not dense_layers or fraction <= 0.0:
        return
    target = max(dense_layers, key=lambda l: int(np.prod(l.kernel.shape)))
    w_b = target.get_weights()
    if not w_b:
        return
    w = w_b[0]
    b = w_b[1] if len(w_b) > 1 else None
    flat = np.abs(w).ravel()
    if flat.size == 0:
        return
    k = int(np.floor(flat.size * fraction))
    if k <= 0:
        return
    thresh = np.partition(flat, k)[k]
    w[np.abs(w) <= thresh] = 0
    target.set_weights([w, b] if b is not None else [w])

# Prepare lightweight pruning strategy (default fraction 0.0 to preserve semantics)
energy_aware_prune(m, fraction=0.0)

# Use built-in checkpoint callback to save best weights by loss (avoids manual tempfile handling)
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='ckpt/best.weights.h5',
    monitor='loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
    verbose=0
)

m.fit(x, y, epochs=3, batch_size=64, verbose=0, callbacks=[checkpoint])
print('done')