import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((28,28,1)); x = layers.Conv2D(8,3,padding='same',activation='relu')(inp)
x = layers.Conv2D(16,3,padding='same',activation='relu')(x); x = layers.GlobalAveragePooling2D()(x); out = layers.Dense(10)(x)
model = keras.Model(inp,out); model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(64,28,28,1).astype('float32'); y = np.random.randint(0,10,(64,))

def prune_most_energy_intensive_conv(model, prune_frac=0.5):
    conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    if not conv_layers:
        return
    target = max(conv_layers, key=lambda l: l.count_params())
    weights = target.get_weights()
    if not weights:
        return
    kernel = weights[0]
    bias = weights[1] if len(weights) > 1 else None
    # L1 norm per output channel (filter)
    filter_norms = np.sum(np.abs(kernel), axis=(0,1,2))
    num_prune = int(np.floor(prune_frac * kernel.shape[3]))
    if num_prune <= 0:
        return
    prune_idx = np.argsort(filter_norms)[:num_prune]
    mask = np.ones_like(kernel)
    for idx in prune_idx:
        mask[..., idx] = 0
    kernel_pruned = kernel * mask
    if bias is not None:
        bias_pruned = bias.copy()
        bias_pruned[prune_idx] = 0
        target.set_weights([kernel_pruned, bias_pruned])
    else:
        target.set_weights([kernel_pruned])

ckpt_cb = keras.callbacks.ModelCheckpoint('model_weights.ckpt', save_weights_only=True, save_freq='epoch', verbose=0)
model.fit(x,y,epochs=2,verbose=0, callbacks=[ckpt_cb])
prune_most_energy_intensive_conv(model, prune_frac=0.5)
print('done')