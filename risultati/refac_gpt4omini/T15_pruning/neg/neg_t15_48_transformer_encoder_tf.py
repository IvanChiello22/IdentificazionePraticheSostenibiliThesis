import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((12, 32))
attn = layers.MultiHeadAttention(4, 8)(inp, inp)
x = layers.Add()([inp, attn])
x = layers.LayerNormalization()(x)

# Pruning the dense layer to reduce energy consumption
ff = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
out = layers.Dense(32)(ff)

model = keras.Model(inp, out)
print(model.count_params())