W = [[1, 2], [3, 4], [5, 6]]
x = [0.5, -1.0]
y = [sum(r[j] * x[j] for j in range(len(x))) for r in W]
print(y)