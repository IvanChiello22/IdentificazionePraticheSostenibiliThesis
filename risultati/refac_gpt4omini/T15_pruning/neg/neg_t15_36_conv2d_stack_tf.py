import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((32, 32, 3))
x = layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
out = layers.Conv2D(32, 3, padding='same')(x)

model = keras.Model(inp, out)
print(model.count_params())