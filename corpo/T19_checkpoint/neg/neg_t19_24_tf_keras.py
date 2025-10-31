# 24) Keras — T19 NEGATIVO: CNN 2D, nessun checkpoint, nessun save_weights
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((28,28,1)); x = layers.Conv2D(8,3,padding='same',activation='relu')(inp)
x = layers.Conv2D(16,3,padding='same',activation='relu')(x); x = layers.GlobalAveragePooling2D()(x); out = layers.Dense(10)(x)
model = keras.Model(inp,out); model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
x = np.random.randn(64,28,28,1).astype('float32'); y = np.random.randint(0,10,(64,))
model.fit(x,y,epochs=2,verbose=0); print('done')
