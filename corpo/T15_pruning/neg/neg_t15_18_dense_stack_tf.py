# 18) DNN denso Keras — T15 NEGATIVO: nessuna sparsificazione
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
model = keras.Sequential([layers.Input((50,)), layers.Dense(128,activation='relu'), layers.Dense(64,activation='relu'), layers.Dense(5)])
model.summary()
