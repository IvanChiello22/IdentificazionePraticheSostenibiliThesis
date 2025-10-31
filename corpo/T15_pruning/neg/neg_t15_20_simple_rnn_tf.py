# 20) SimpleRNN — T15 NEGATIVO: nessuna riduzione dei parametri
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
model = keras.Sequential([layers.Input((10,8)), layers.SimpleRNN(32), layers.Dense(3)])
model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(4,10,8).astype('float32'); y = np.random.randint(0,3,(4,))
model.train_on_batch(x,y); print("done")
