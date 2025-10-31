# 47) ConvLSTM2D — T15 NEGATIVO: nessuna riduzione dei filtri
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((5, 16, 16, 1))
x = layers.ConvLSTM2D(8, 3, padding='same', return_sequences=False)(inp)
out = layers.Conv2D(1, 1)(x)
model = keras.Model(inp,out); print(model.count_params())
