import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(7, 16)),
    layers.GRU(24, return_sequences=True),
    layers.GRU(24),
    layers.Dense(4)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

x = np.random.randn(5, 7, 16).astype('float32')
y = np.random.randint(0, 4, (5,))

# Aggiunta di checkpoint per il salvataggio del modello durante l'allenamento
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

model.fit(x, y, epochs=10, callbacks=[checkpoint_callback], verbose=0)
print("done")