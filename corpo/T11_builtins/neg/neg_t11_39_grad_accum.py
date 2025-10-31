# 39) Accumulate grad (somma manuale)
acc = [0.0, 0.0, 0.0]
grads_per_step = [[0.1, -0.2, 0.3],
                  [0.05, 0.02, -0.01],
                  [0.2, -0.1, 0.0]]
for g in grads_per_step:
    for i in range(len(acc)):
        acc[i] += g[i]
print(acc)
