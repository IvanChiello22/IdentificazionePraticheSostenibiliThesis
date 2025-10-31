# 37) Stack LSTM — T15 NEGATIVO: nessuna potatura
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
model = keras.Sequential([layers.Input((10,16)), layers.LSTM(32, return_sequences=True), layers.LSTM(32), layers.Dense(4)])
print(model.count_params())
