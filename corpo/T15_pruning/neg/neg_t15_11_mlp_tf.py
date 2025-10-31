# 11) MLP Keras — T15 NEGATIVO: nessun pruning (no TF-MOT)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(20,)), layers.Dense(3)])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
import numpy as np
x = np.random.randn(32,20).astype('float32'); y = np.random.randint(0,3,(32,))
model.train_on_batch(x,y)
print("done")
