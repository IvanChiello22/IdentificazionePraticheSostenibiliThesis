# 33) Embedding + media — T15 NEGATIVO: nessuna riduzione dimensionale
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input(shape=(10,))
x = layers.Embedding(100, 32)(inp)
x = tf.reduce_mean(x, axis=1)
out = layers.Dense(5)(x)
model = keras.Model(inp,out); print(model.count_params())
