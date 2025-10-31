# T19 POS (energy/efficiency-aware checkpointing) — snippet 23
# T19 POS — Keras: ModelCheckpoint save_best_only + save_weights_only (meno I/O)
import tensorflow as tf, numpy as np, os
from tensorflow import keras; from tensorflow.keras import layers
os.makedirs('ckpt', exist_ok=True)
m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
cb = keras.callbacks.ModelCheckpoint('ckpt/best.weights.h5', monitor='loss', save_best_only=True, save_weights_only=True, mode='min')
x = np.random.randn(256,20).astype('float32'); y = np.random.randint(0,3,(256,))
m.fit(x,y,epochs=3,batch_size=64,callbacks=[cb],verbose=0); print('done')