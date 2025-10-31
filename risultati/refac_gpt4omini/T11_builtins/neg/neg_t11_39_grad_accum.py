acc = [0.0, 0.0, 0.0]
grads_per_step = [[0.1, -0.2, 0.3],
                  [0.05, 0.02, -0.01],
                  [0.2, -0.1, 0.0]]
acc = [sum(x) for x in zip(*grads_per_step)]
print(acc)