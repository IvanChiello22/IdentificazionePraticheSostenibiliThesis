import tensorflow as tf

ds = tf.data.Dataset.from_tensor_slices((tf.random.normal([256, 10]), tf.random.uniform([256], maxval=2, dtype=tf.int32))).batch(32)

m = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(2)
])

m.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', save_weights_only=True, save_best_only=True)

m.fit(ds, epochs=2, verbose=0, callbacks=[checkpoint_callback])

print('done')