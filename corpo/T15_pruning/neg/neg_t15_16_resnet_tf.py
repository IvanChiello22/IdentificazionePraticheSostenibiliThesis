# 16) Mini-ResNet Keras — T15 NEGATIVO: nessun canale eliminato
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input(shape=(32,32,3))
x = layers.Conv2D(16,3,padding='same',activation='relu')(inp)
r = layers.Conv2D(16,3,padding='same')(x)
x = layers.Add()([x,r]); x = layers.Activation('relu')(x)
x = layers.GlobalAveragePooling2D()(x)
out = layers.Dense(10)(x)
model = keras.Model(inp,out); print(model.output_shape)
