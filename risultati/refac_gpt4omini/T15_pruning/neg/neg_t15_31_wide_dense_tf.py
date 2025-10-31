import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((256,))
x = layers.Dense(256, activation='relu')(inp)
x = layers.Dense(128, activation='relu')(x)
out = layers.Dense(10)(x)

model = keras.Model(inp, out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()