import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((28, 28, 1))
x = layers.Conv2D(8, 3, padding='same', activation='relu')(inp)
z = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
u = layers.UpSampling2D()(z)
out = layers.Conv2D(1, 3, padding='same')(u)

model = keras.Model(inp, out)

# Implementing checkpointing
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

# Training the model with checkpointing
model.compile(optimizer='adam', loss='mse')
# Assuming 'train_data' and 'train_labels' are defined
model.fit(train_data, train_labels, epochs=10, callbacks=[checkpoint_callback])

print(model.count_params())