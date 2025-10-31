# 7) ReLU elemento-per-elemento
x = [-2.0, -0.5, 0.0, 1.2, 3.4]
y = [max(v, 0.0) for v in x]
print(y)