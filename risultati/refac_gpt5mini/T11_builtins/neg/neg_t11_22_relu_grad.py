# 22) ReLU grad
x = [-1.0, 0.5, 3.0]
grad = [1.0 if v > 0 else 0.0 for v in x]
print(grad)