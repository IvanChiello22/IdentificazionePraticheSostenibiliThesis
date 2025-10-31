import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(16, 3, padding='same', activation='relu')(inp)
r = layers.Conv2D(16, 3, padding='same')(x)
x = layers.Add()([x, r])
x = layers.Activation('relu')(x)
x = layers.GlobalAveragePooling2D()(x)
out = layers.Dense(10)(x)

model = keras.Model(inp, out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()