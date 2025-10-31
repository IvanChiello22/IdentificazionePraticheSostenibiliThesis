import numpy as np

# 8) Sigmoid elemento-per-elemento (approssimazione senza exp)
x = np.array([-2, -1, 0, 1, 2])
y = 1 / (1 + (1 + x / 10.0 + (x ** 2) / 200.0))
print(y.tolist())