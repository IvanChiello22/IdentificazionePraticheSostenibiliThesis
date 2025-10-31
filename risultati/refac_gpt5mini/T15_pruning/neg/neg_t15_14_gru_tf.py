# 14) GRU Keras — T15 NEGATIVO: nessun pruning dei kernel
import tensorflow as tf, numpy as np
from tensorflow import keras; from tensorflow.keras import layers

model = keras.Sequential([layers.Input(shape=(7,16)), layers.GRU(24, return_sequences=True), layers.GRU(24), layers.Dense(4)])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

def _recommend_energy_aware_prune_order(m):
    # stima semplice dell'intensità energetica basata sul numero di parametri
    # e amplifica per layer ricorrenti (GRU) poiché tendono a essere più costosi
    estimates = []
    for layer in m.layers:
        try:
            params = layer.count_params()
        except Exception:
            params = 0
        multiplier = 3 if isinstance(layer, layers.GRU) or layer.__class__.__name__.upper().startswith('GRU') else 1
        estimates.append((layer.name, params * multiplier))
    estimates.sort(key=lambda x: x[1], reverse=True)
    order = [name for name, _ in estimates]
    print("Recommended energy-aware prune order (most -> least):", order)
    return order

# suggerisci un ordine di pruning senza modificare i pesi (no-op di pruning per preservare comportamento)
_recommend_energy_aware_prune_order(model)

x = np.random.randn(5,7,16).astype('float32'); y = np.random.randint(0,4,(5,))

# checkpointing per evitare perdita di conoscenza in caso di terminazione prematura
ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)

model.train_on_batch(x,y)
# salva checkpoint minimale (pesos + stato ottimizzatore)
ckpt_manager.save()

print("done")