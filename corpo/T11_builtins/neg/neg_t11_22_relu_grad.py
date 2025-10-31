# 22) ReLU grad
x = [-1.0, 0.5, 3.0]
grad = []
for v in x:
    grad.append(1.0 if v > 0 else 0.0)
print(grad)
