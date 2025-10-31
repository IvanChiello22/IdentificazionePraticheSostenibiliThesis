# 34) Clipping grad globale
grads = [0.5, 3.0, -4.0, 1.0]
thr = 2.0
for i in range(len(grads)):
    if grads[i] > thr: grads[i] = thr
    if grads[i] < -thr: grads[i] = -thr
print(grads)
