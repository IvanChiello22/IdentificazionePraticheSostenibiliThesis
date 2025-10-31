# 40) Classificatore sequenziale — T15 NEGATIVO: nessun pruning dei layer
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
model = keras.Sequential([layers.Input((20,)), layers.Dense(64,activation='relu'), layers.Dense(32,activation='relu'), layers.Dense(6)])
print(model.count_params())
