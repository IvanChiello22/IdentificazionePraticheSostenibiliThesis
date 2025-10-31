# 12) CNN Keras — T15 NEGATIVO: nessun pruning strutturale
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input(shape=(28,28,1))
x = layers.Conv2D(8,3,padding='same',activation='relu')(inp)
x = layers.Conv2D(16,3,padding='same',activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
out = layers.Dense(10)(x)
model = keras.Model(inp,out)
optimizer = keras.optimizers.SGD()
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.train_on_batch(np.random.randn(8,28,28,1).astype('float32'), np.random.randint(0,10,(8,)))
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=1)
ckpt_manager.save()
print("done")