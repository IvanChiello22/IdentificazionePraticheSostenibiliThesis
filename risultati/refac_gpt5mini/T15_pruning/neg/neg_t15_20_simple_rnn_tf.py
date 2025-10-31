# 20) SimpleRNN — T15 NEGATIVO: nessuna riduzione dei parametri
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
model = keras.Sequential([layers.Input((10,8)), layers.SimpleRNN(32), layers.Dense(3)])
model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
checkpoint_dir = './checkpoints'
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
def _get_most_param_intensive_layer(keras_model):
    layers_list = [l for l in keras_model.layers if hasattr(l, 'count_params')]
    if not layers_list:
        return None
    return max(layers_list, key=lambda l: l.count_params())
def _prune_layer_by_magnitude(layer, fraction):
    if fraction <= 0.0:
        return
    weights = layer.get_weights()
    if not weights:
        return
    pruned_weights = []
    for w in weights:
        flat = w.ravel()
        k = int(flat.size * fraction)
        if k <= 0:
            pruned_weights.append(w)
            continue
        threshold = np.partition(np.abs(flat), k)[k]
        mask = np.abs(w) > threshold
        pruned_weights.append(w * mask)
    layer.set_weights(pruned_weights)
most_heavy = _get_most_param_intensive_layer(model)
prune_fraction = 0.0
if most_heavy is not None:
    _prune_layer_by_magnitude(most_heavy, prune_fraction)
x = np.random.randn(4,10,8).astype('float32'); y = np.random.randint(0,3,(4,))
model.train_on_batch(x,y); manager.save(); print("done")