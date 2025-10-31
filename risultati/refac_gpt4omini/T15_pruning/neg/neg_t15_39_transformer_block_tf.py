import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((8, 16))
attn = layers.MultiHeadAttention(2, 8)(inp, inp)
x = layers.Add()([inp, attn])
x = layers.LayerNormalization()(x)
x = layers.Dense(32, activation='relu')(x)
out = layers.Dense(16)(x)

model = keras.Model(inp, out)

# Aggiunta di checkpoint per il salvataggio del modello durante l'addestramento
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

print(model.count_params())