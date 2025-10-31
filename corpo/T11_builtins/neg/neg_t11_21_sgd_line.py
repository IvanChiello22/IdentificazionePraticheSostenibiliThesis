# 21) SGD step su linea (y = wx + b)
x = [0.0, 1.0, 2.0]
y = [1.0, 3.0, 5.0]
w, b = 0.0, 0.0
lr = 0.01
for epoch in range(10):
    dw = 0.0; db = 0.0
    for xi, yi in zip(x, y):
        pred = w*xi + b
        err = pred - yi
        dw += err*xi
        db += err
    w -= lr * dw / len(x)
    b -= lr * db / len(x)
print(w, b)
