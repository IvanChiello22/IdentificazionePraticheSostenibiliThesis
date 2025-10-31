# 18) One-hot encoding
labels = [0, 2, 1]
num_classes = 3
oh = []
for y in labels:
    v = [0]*num_classes
    v[y] = 1
    oh.append(v)
print(oh)
