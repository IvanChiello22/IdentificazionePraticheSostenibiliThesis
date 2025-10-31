# 35) BiRNN — T15 NEGATIVO: parametri completi
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((12,16)); x = layers.Bidirectional(layers.SimpleRNN(24, return_sequences=True))(inp); x = layers.Bidirectional(layers.SimpleRNN(24))(x); out = layers.Dense(3)(x)
model = keras.Model(inp,out); print(model.count_params())
