# 19) Autoencoder — T15 NEGATIVO: nessun pruning del bottleneck
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input(shape=(32,))
z = layers.Dense(16, activation='relu')(inp)
z = layers.Dense(8, activation='relu')(z)
x = layers.Dense(16, activation='relu')(z)
out = layers.Dense(32)(x)
model = keras.Model(inp, out)
print(sum([w.numpy().size if hasattr(w,'numpy') else w.shape.num_elements() if hasattr(w.shape,'num_elements') else 0 for w in model.weights]))
