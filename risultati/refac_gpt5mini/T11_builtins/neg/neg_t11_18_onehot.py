# 18) One-hot encoding
labels = [0, 2, 1]
num_classes = 3
oh = [[1 if i == y else 0 for i in range(num_classes)] for y in labels]
print(oh)