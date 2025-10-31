# T19 POS (energy/efficiency-aware checkpointing) — snippet 39
# T19 POS — Keras: ModelCheckpoint save_best_only + save_weights_only (meno I/O)
import tensorflow as tf, numpy as np, os
from tensorflow import keras; from tensorflow.keras import layers
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
cb = keras.callbacks.ModelCheckpoint('ckpt/best.weights.h5', monitor='loss', save_best_only=True, save_weights_only=True, mode='min')
x = np.random.randn(256,20).astype('float32'); y = np.random.randint(0,3,(256,))
m.fit(x,y,epochs=3,batch_size=64,callbacks=[cb],verbose=0); print('done')
# Energy-aware post-training pruning (prune most parameter-heavy layer in saved checkpoint)
_ckpt = 'ckpt/best.weights.h5'
if os.path.exists(_ckpt):
    pruned_model = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
    # load weights from checkpoint into fresh model
    try:
        pruned_model.load_weights(_ckpt)
        # identify most parameter-heavy layer
        layer_param_counts = [(idx, sum(w.size for w in layer.get_weights())) for idx, layer in enumerate(pruned_model.layers)]
        if layer_param_counts:
            idx_max = max(layer_param_counts, key=lambda t: t[1])[0]
            layer = pruned_model.layers[idx_max]
            weights = layer.get_weights()
            if weights:
                kernel = weights[0]
                # prune smallest-magnitude weights by a small fraction (energy-aware)
                prune_fraction = 0.2
                flat_abs = np.abs(kernel).ravel()
                if flat_abs.size:
                    thresh = np.percentile(flat_abs, prune_fraction * 100)
                    mask = np.abs(kernel) > thresh
                    kernel_pruned = kernel * mask
                    weights[0] = kernel_pruned
                    layer.set_weights(weights)
                    pruned_model.save_weights(_ckpt)
    except Exception:
        pass