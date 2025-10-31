# 33) Embedding + media — T15 NEGATIVO: nessuna riduzione dimensionale
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input(shape=(10,))
x = layers.Embedding(100, 32)(inp)
x = layers.GlobalAveragePooling1D()(x)
out = layers.Dense(5)(x)
model = keras.Model(inp,out)
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', save_weights_only=True)
print(model.count_params())