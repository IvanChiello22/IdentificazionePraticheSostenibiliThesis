x = [1, 3, 2, 5, 4, 6]
k = 2
pooled = [max(x[i:i+k]) for i in range(0, len(x), k)]
print(pooled)