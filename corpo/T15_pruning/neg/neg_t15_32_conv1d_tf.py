# 32) Conv1D — T15 NEGATIVO: canali completi
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((100,4)); x = layers.Conv1D(16,5,padding='same',activation='relu')(inp); x = layers.Conv1D(32,3,padding='same')(x); model = keras.Model(inp,x)
print(model.output_shape)
