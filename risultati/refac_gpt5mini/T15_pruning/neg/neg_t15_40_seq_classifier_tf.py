# 40) Classificatore sequenziale — T15 NEGATIVO: nessun pruning dei layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot

prune = tfmot.sparsity.keras.prune_low_magnitude
schedule_high = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.6, begin_step=0, end_step=1000)
schedule_low = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.3, begin_step=0, end_step=1000)

model = keras.Sequential([
    layers.Input((20,)),
    prune(layers.Dense(64, activation='relu'), pruning_schedule=schedule_high),
    prune(layers.Dense(32, activation='relu'), pruning_schedule=schedule_low),
    layers.Dense(6)
])

checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath='checkpoint.h5', save_weights_only=True)
print(model.count_params())