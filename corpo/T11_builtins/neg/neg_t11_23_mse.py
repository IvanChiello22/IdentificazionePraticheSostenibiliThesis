# 23) MSE loss
y_true = [1.0, 2.0, 3.0]
y_pred = [0.8, 2.5, 2.9]
s = 0.0
for a, b in zip(y_true, y_pred):
    d = a - b
    s += d*d
mse = s / len(y_true)
print(mse)
