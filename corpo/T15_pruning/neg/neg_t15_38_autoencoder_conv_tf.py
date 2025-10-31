# 38) Autoencoder conv — T15 NEGATIVO: nessun pruning
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((28,28,1))
x = layers.Conv2D(8,3,padding='same',activation='relu')(inp)
z = layers.Conv2D(16,3,padding='same',activation='relu')(x)
u = layers.UpSampling2D()(z)
out = layers.Conv2D(1,3,padding='same')(u)
model = keras.Model(inp,out); print(model.count_params())
