# 46) TimeDistributed Dense — T15 NEGATIVO: layer completi
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((10, 32))
x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(inp)
out = layers.TimeDistributed(layers.Dense(5))(x)
model = keras.Model(inp,out); print(model.count_params())
model.compile(optimizer='adam', loss='mse')
x_train = tf.random.normal((4, 10, 32))
y_train = tf.random.normal((4, 10, 5))
checkpoint_cb = keras.callbacks.ModelCheckpoint('checkpoint.h5', save_weights_only=True)
model.fit(x_train, y_train, epochs=1, callbacks=[checkpoint_cb], verbose=0)