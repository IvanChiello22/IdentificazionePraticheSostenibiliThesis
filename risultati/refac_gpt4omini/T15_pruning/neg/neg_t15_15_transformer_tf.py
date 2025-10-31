import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input(shape=(6, 32))
attn = layers.MultiHeadAttention(num_heads=4, key_dim=8)(inp, inp)
x = layers.Add()([inp, attn])
x = layers.LayerNormalization()(x)
ff = layers.Dense(64, activation='relu')(x)
out = layers.Dense(32)(ff)
model = keras.Model(inputs=inp, outputs=out)

# Utilizzo di checkpoint durante l'addestramento
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

# Simulazione di addestramento con checkpoint
model.compile(optimizer='adam', loss='mse')
model.fit(np.random.randn(100, 6, 32).astype('float32'), 
          np.random.randn(100, 32).astype('float32'), 
          epochs=10, 
          callbacks=[checkpoint_callback])

y = model(np.random.randn(3, 6, 32).astype('float32'))
print(y.shape)