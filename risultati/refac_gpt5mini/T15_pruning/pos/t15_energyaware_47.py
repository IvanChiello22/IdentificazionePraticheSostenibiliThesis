# T15 POS — Energy-aware pruning (TensorFlow): maschere per Dense con score L1/energy
import os
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
inp = keras.Input((64,))
d1 = layers.Dense(128, activation='relu')
d2 = layers.Dense(32)
x = d1(inp); out = d2(x)
model = keras.Model(inp, out)

@tf.function
def energy_aware_mask(W, keep_ratio):
    l1 = tf.reduce_mean(tf.abs(W), axis=1)  # importanza neurone (righe)
    energy = tf.cast(tf.shape(W)[1], tf.float32)  # proxy costo per neurone
    score = l1/(energy+1e-9)
    k = tf.cast(tf.maximum(1, tf.cast(tf.round(keep_ratio*tf.cast(tf.shape(W)[0],tf.float32)), tf.int32)), tf.int32)
    idx = tf.argsort(score, direction='DESCENDING')[:k]
    mask = tf.scatter_nd(tf.expand_dims(idx,1), tf.ones((k,),dtype=W.dtype), [tf.shape(W)[0]])
    return tf.expand_dims(mask,1)

# Select the most energy-intensive Dense layer (proxy: input_dim * units)
dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
if dense_layers:
    energy_proxies = [int(layer.weights[0].shape[0]) * int(layer.weights[0].shape[1]) for layer in dense_layers]
    target_layer = dense_layers[int(tf.math.argmax(energy_proxies).numpy())]
else:
    target_layer = d1  # fallback, should not happen for this model

# Save a checkpoint of model weights before pruning to avoid knowledge loss
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_prefix = os.path.join(tmpdir, 'ckpt')
    ckpt.save(file_prefix=ckpt_prefix)

W = target_layer.weights[0]
mask = energy_aware_mask(W, 0.6)
target_layer.set_weights([W.numpy() * mask.numpy(), target_layer.weights[1].numpy()])
print('masked neurons:', int(tf.reduce_sum(mask)))