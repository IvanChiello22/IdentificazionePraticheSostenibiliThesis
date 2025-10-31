# 34) Clipping grad globale
grads = [0.5, 3.0, -4.0, 1.0]
thr = 2.0
grads = [max(min(g, thr), -thr) for g in grads]
print(grads)