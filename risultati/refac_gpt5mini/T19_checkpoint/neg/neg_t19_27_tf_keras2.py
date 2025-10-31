import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
model = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
# Energy-aware pruning: prune a small fraction of the smallest-magnitude weights from the most parameter-heavy layer
prune_fraction = 0.1
# Identify the layer with the largest number of kernel parameters
def _kernel_size(layer):
    w = layer.get_weights()
    return w[0].size if w else 0
target_layer = max(model.layers, key=_kernel_size)
weights = target_layer.get_weights()
if weights:
    kernel, bias = weights[0], weights[1] if len(weights) > 1 else None
    flat_abs = np.abs(kernel).flatten()
    k = int(np.floor(prune_fraction * flat_abs.size))
    if k > 0:
        thresh = np.partition(flat_abs, k)[k]
        mask = np.abs(kernel) > thresh
        pruned_kernel = kernel * mask
        new_weights = [pruned_kernel] + ([bias] if bias is not None else [])
        target_layer.set_weights(new_weights)
# Use ModelCheckpoint callback to save weights during training
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath='ckpt-{epoch}', save_weights_only=True, save_freq='epoch')
x = np.random.randn(256,20).astype('float32'); y = np.random.randint(0,3,(256,))
model.fit(x, y, epochs=3, batch_size=32, verbose=0, callbacks=[checkpoint_cb]); print('done')