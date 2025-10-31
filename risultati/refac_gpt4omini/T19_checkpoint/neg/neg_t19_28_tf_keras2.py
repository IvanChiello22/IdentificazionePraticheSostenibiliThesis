import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers

inp = keras.Input((12,10))
x = layers.LSTM(16)(inp)
out = layers.Dense(2)(x)

model = keras.Model(inp, out)
model.compile(optimizer='sgd', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

x = np.random.randn(64,12,10).astype('float32')
y = np.random.randint(0,2,(64,))

checkpoint_callback = keras.callbacks.ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
model.fit(x, y, epochs=3, batch_size=16, verbose=0, callbacks=[checkpoint_callback])

print('done')