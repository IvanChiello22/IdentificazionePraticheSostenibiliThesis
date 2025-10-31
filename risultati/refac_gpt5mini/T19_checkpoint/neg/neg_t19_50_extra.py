import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices((tf.random.normal([256,10]), tf.random.uniform([256], maxval=2, dtype=tf.int32))).batch(32).prefetch(tf.data.AUTOTUNE)
m = tf.keras.Sequential([tf.keras.layers.Dense(32,activation='relu',input_shape=(10,)), tf.keras.layers.Dense(2)])
m.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint.ckpt', save_weights_only=True, save_freq='epoch')
m.fit(ds, epochs=2, verbose=0, callbacks=[checkpoint_cb]); print('done')