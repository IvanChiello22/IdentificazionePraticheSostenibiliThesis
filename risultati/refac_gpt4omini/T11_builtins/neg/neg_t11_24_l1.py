y_true = [0, 1, 2]
y_pred = [0.2, 0.9, 2.5]
loss = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
print(loss)