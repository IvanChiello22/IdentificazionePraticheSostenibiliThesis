# 39) Accumulate grad (somma manuale)
acc = [0.0, 0.0, 0.0]
grads_per_step = [[0.1, -0.2, 0.3],
                  [0.05, 0.02, -0.01],
                  [0.2, -0.1, 0.0]]
acc = [acc[i] + sum(g[i] for g in grads_per_step) for i in range(len(acc))]
print(acc)