# 13) LSTM Keras — T15 NEGATIVO: layer completi, nessun pruning
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
model = keras.Sequential([layers.Input(shape=(12,10)), layers.LSTM(20), layers.Dense(2)])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(4,12,10).astype('float32'); y = np.random.randint(0,2,(4,))
model.train_on_batch(x,y); print("done")
