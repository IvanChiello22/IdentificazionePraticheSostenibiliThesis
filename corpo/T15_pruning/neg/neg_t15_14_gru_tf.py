# 14) GRU Keras — T15 NEGATIVO: nessun pruning dei kernel
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
model = keras.Sequential([layers.Input(shape=(7,16)), layers.GRU(24, return_sequences=True), layers.GRU(24), layers.Dense(4)])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(5,7,16).astype('float32'); y = np.random.randint(0,4,(5,))
model.train_on_batch(x,y); print("done")
