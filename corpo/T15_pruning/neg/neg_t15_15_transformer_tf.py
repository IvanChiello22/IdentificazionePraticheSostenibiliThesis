# 15) MHA+FFN Keras — T15 NEGATIVO: nessuna rimozione di head/neuroni
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input(shape=(6,32))
attn = layers.MultiHeadAttention(num_heads=4, key_dim=8)(inp, inp)
x = layers.Add()([inp, attn])
x = layers.LayerNormalization()(x)
ff = layers.Dense(64, activation='relu')(x)
out = layers.Dense(32)(ff)
model = keras.Model(inp, out)
y = model(np.random.randn(3,6,32).astype('float32')); print(y.shape)
