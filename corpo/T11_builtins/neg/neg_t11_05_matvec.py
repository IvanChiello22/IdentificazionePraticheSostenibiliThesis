# 5) Prodotto matrice-vettore
W = [[1, 2], [3, 4], [5, 6]]
x = [0.5, -1.0]
y = []
for r in W:
    s = 0.0
    for j in range(len(x)):
        s += r[j] * x[j]
    y.append(s)
print(y)
