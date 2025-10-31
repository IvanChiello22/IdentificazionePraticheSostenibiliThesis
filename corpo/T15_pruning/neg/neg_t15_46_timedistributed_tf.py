# 46) TimeDistributed Dense — T15 NEGATIVO: layer completi
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((10, 32))
x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(inp)
out = layers.TimeDistributed(layers.Dense(5))(x)
model = keras.Model(inp,out); print(model.count_params())
