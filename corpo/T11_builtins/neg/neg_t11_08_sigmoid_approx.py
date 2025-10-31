# 8) Sigmoid elemento-per-elemento (approssimazione senza exp)
x = [-2, -1, 0, 1, 2]
y = []
for v in x:
    # approssimazione semplice della sigmoid
    e = 1 + (1 + v/10.0 + (v*v)/200.0)  # artificiosa, solo per evitare librerie
    s = 1.0 / e
    y.append(s)
print(y)
