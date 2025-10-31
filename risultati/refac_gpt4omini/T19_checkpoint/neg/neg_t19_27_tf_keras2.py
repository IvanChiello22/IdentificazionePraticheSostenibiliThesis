import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers, callbacks

model = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

x = np.random.randn(256,20).astype('float32'); y = np.random.randint(0,3,(256,))

checkpoint = callbacks.ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
model.fit(x, y, epochs=3, batch_size=32, verbose=0, callbacks=[checkpoint])

print('done')