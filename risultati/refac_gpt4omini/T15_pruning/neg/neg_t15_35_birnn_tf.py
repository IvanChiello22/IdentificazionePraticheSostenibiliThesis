import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inp = keras.Input((12, 16))
x = layers.Bidirectional(layers.SimpleRNN(24, return_sequences=True))(inp)
x = layers.Bidirectional(layers.SimpleRNN(24))(x)
out = layers.Dense(3)(x)

model = keras.Model(inp, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint_callback = keras.callbacks.ModelCheckpoint('model_checkpoint.h5', save_best_only=True)

# Assuming you have training data in `x_train` and `y_train`
model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint_callback])
print(model.count_params())