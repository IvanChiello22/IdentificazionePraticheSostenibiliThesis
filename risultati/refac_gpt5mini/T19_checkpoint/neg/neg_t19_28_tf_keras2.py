import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
inp = keras.Input((12,10))
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
end_step = int(np.ceil(64/16) * 3)
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=end_step
    )
}
x = prune_low_magnitude(layers.LSTM(16), **pruning_params)(inp)
out = layers.Dense(2)(x)
model = keras.Model(inp,out)
model.compile(optimizer='sgd', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(64,12,10).astype('float32'); y = np.random.randint(0,2,(64,))
checkpoint_cb = keras.callbacks.ModelCheckpoint('model.ckpt', save_weights_only=True, save_freq='epoch', verbose=0)
model.fit(x,y,epochs=3,batch_size=16,verbose=0, callbacks=[checkpoint_cb, tfmot.sparsity.keras.UpdatePruningStep()])
print('done')