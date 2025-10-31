# 39) Blocco Transformer — T15 NEGATIVO: nessuna rimozione strutturale
import tensorflow as tf
from tensorflow import keras; from tensorflow.keras import layers
inp = keras.Input((8,16)); attn = layers.MultiHeadAttention(2,8)(inp,inp); x = layers.Add()([inp,attn]); x = layers.LayerNormalization()(x); x = layers.Dense(32,activation='relu')(x); out = layers.Dense(16)(x)
model = keras.Model(inp,out); print(model.count_params())
