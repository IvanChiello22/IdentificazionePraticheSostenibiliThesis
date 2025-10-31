# 16) MaxPool 1D
x = [1, 3, 2, 5, 4, 6]
k = 2
pooled = []
for i in range(0, len(x), k):
    m = x[i]
    for j in range(1, k):
        if i + j < len(x) and x[i+j] > m:
            m = x[i+j]
    pooled.append(m)
print(pooled)
