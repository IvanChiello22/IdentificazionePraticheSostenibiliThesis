# 50) TensorFlow — T19 NEGATIVO: tf.data + Keras fit senza callback
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices((tf.random.normal([256,10]), tf.random.uniform([256], maxval=2, dtype=tf.int32))).batch(32)
m = tf.keras.Sequential([tf.keras.layers.Dense(32,activation='relu',input_shape=(10,)), tf.keras.layers.Dense(2)])
m.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
m.fit(ds, epochs=2, verbose=0); print('done')
