# 24) L1 loss
y_true = [0,1,2]
y_pred = [0.2, 0.9, 2.5]
s = sum(abs(a - b) for a, b in zip(y_true, y_pred))
print(s/len(y_true))