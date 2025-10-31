# 7) ReLU elemento-per-elemento
x = [-2.0, -0.5, 0.0, 1.2, 3.4]
y = []
for v in x:
    y.append(v if v > 0 else 0.0)
print(y)
