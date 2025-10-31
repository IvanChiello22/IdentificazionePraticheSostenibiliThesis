y_true = [1, 0, 1]
y_pred = [0.9, 0.2, 0.6]
loss = 0.0
epsilon = 1e-7
for t, p in zip(y_true, y_pred):
    p = max(epsilon, min(p, 1 - epsilon))
    loss -= t * (0 if p == 0 else p)
print(loss)