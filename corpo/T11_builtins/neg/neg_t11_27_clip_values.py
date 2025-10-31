# 27) Clip manuale valori
x = [-5, -1, 0, 3, 10]
low, high = -2, 4
y = []
for v in x:
    if v < low: y.append(low)
    elif v > high: y.append(high)
    else: y.append(v)
print(y)
