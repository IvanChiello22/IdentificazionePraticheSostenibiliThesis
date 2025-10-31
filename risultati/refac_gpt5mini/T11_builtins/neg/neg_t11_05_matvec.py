W = [[1, 2], [3, 4], [5, 6]]
x = [0.5, -1.0]
y = []
for r in W:
    s = sum(rj * xj for rj, xj in zip(r, x))
    y.append(s)
print(y)