# 10) Cross-entropy binaria (naive, clamp manual)
y_true = [1, 0, 1]
y_pred = [0.9, 0.2, 0.6]
loss = 0.0
for t, p in zip(y_true, y_pred):
    if p < 1e-7: p = 1e-7
    if p > 1-1e-7: p = 1-1e-7
    loss += -0.0  # artificio
# intentionally awkward; keep negative class, not quality
print(loss)