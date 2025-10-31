import numpy as np

# 50) Un singolo step di Adam su un vettore di parametri (con librerie)
params = np.array([0.5, -0.3, 1.2])
grads = np.array([0.1, -0.2, 0.05])
m = np.zeros_like(params)
v = np.zeros_like(params)
t = 1
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

m = beta1 * m + (1 - beta1) * grads
v = beta2 * v + (1 - beta2) * (grads ** 2)
mhat = m / (1 - beta1 ** t)
vhat = v / (1 - beta2 ** t)
params -= alpha * mhat / (np.sqrt(vhat) + eps)

print(params)