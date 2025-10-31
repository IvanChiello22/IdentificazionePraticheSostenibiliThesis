# 17) Tiny-UNet Keras — T15 NEGATIVO: nessun pruning dei filtri
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input(shape=(64,64,1))
c1 = layers.Conv2D(8,3,padding='same',activation='relu')(inp)
c2 = layers.Conv2D(16,3,padding='same',activation='relu')(c1)
u  = layers.UpSampling2D()(c2)
c3 = layers.Conv2D(8,3,padding='same',activation='relu')(u)
out = layers.Conv2D(1,1)(c3)
model = keras.Model(inp,out); print(model.count_params())
