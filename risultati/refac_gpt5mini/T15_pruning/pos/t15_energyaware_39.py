# T15 POS — Energy-aware pruning (TensorFlow): maschere per Dense con score L1/energy
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

def most_energy_intensive_layer(keras_model):
    candidate = None
    max_energy = -1
    for layer in keras_model.layers:
        if layer.weights:
            kernel = layer.weights[0]
            elems = kernel.shape.num_elements()
            if elems is None:
                elems = int(tf.size(kernel).numpy())
            if elems > max_energy:
                max_energy = elems
                candidate = layer
    return candidate

# checkpoint per evitare perdita di conoscenza in caso di terminazione prematura
ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)
manager.save()

selected_layer = most_energy_intensive_layer(model)
W = selected_layer.weights[0]
mask = energy_aware_mask(W, 0.6)
selected_layer.set_weights([W.numpy() * mask.numpy(), selected_layer.weights[1].numpy()])
print('masked neurons:', int(tf.reduce_sum(mask)))