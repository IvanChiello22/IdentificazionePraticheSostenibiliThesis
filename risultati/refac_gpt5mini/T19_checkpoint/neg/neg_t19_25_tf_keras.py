# 25) Keras — T19 NEGATIVO: GRU seq2label senza checkpoint
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((15,16)); x = layers.GRU(24)(inp); out = layers.Dense(4)(x)
m = keras.Model(inp,out); m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(80,15,16).astype('float32'); y = np.random.randint(0,4,(80,))
cp = keras.callbacks.ModelCheckpoint('model.ckpt', save_weights_only=True, save_freq='epoch')
m.fit(x,y,epochs=2,verbose=0, callbacks=[cp]); print('done')