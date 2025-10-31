import math
import pickle
import time

def save_checkpoint(state, filename=None):
    if filename is None:
        filename = f'checkpoint_{int(time.time())}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(state, f)

def energy_aware_prune(params, grads, enabled=False, prune_fraction=0.0):
    if not enabled or prune_fraction <= 0.0:
        return params
    energies = [abs(p * g) for p, g in zip(params, grads)]
    k = int(len(params) * prune_fraction)
    if k <= 0:
        return params
    idx_sorted = sorted(range(len(energies)), key=lambda i: energies[i], reverse=True)
    pruned = params.copy()
    for i in idx_sorted[:k]:
        pruned[i] = 0.0
    return pruned

params = [0.5, -0.3, 1.2]
grads =  [0.1, -0.2, 0.05]
m = [0.0, 0.0, 0.0]
v = [0.0, 0.0, 0.0]
t = 1
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

save_checkpoint({'params': params.copy(), 'm': m.copy(), 'v': v.copy(), 't': t}, filename='checkpoint_before.pkl')

params = energy_aware_prune(params, grads, enabled=False, prune_fraction=0.1)

for i in range(len(params)):
    m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
    v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] * grads[i])
    mhat = m[i] / (1 - beta1**t)
    vhat = v[i] / (1 - beta2**t)
    params[i] -= alpha * mhat / (math.sqrt(vhat) + eps)

save_checkpoint({'params': params.copy(), 'm': m.copy(), 'v': v.copy(), 't': t}, filename='checkpoint_after.pkl')

print(params)