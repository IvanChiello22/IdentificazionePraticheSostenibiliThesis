grads = [0.5, 3.0, -4.0, 1.0]
thr = 2.0
grads = [max(min(grad, thr), -thr) for grad in grads]
print(grads)