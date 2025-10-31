import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((6, 32))
attn = layers.MultiHeadAttention(num_heads=4, key_dim=8)(inp, inp)
model = keras.Model(inputs=inp, outputs=attn)
print(model.output_shape)