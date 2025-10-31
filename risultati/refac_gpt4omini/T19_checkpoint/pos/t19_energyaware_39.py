import tensorflow as tf, numpy as np, os
from tensorflow import keras; from tensorflow.keras import layers

os.makedirs('ckpt', exist_ok=True)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(3)
])

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'ckpt/best.weights.h5',
    monitor='loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min'
)

x = np.random.randn(256, 20).astype('float32')
y = np.random.randint(0, 3, (256,))

model.fit(x, y, epochs=3, batch_size=64, callbacks=[checkpoint_callback], verbose=0)

print('done')